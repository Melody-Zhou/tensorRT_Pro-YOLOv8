#include "yolo_depth.hpp"
#include <atomic>
#include <mutex>
#include <queue>
#include <condition_variable>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/preprocess_kernel.cuh>
#include <common/monopoly_allocator.hpp>
#include <common/cuda_tools.hpp>

namespace YoloDepth{
    using namespace cv;
    using namespace std;

    void resize_depth_align_corners_invoker(
        const float* src, int src_width, int src_height,
        float* dst, int dst_width, int dst_height,
        cudaStream_t stream
    );

    struct ImageInfo{
        int width  = 0;
        int height = 0;
    };

    using ControllerImpl = InferController
    <
        Mat,                    // input
        Mat,                    // output
        tuple<string, int>,     // start param
        ImageInfo               // additional, per job source size
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:

        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            stop();
        }

        virtual bool startup(
            const string& file, int gpuid,
            bool use_multi_preprocess_stream
        ){
            normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
            use_multi_preprocess_stream_ = use_multi_preprocess_stream;
            return ControllerImpl::startup(make_tuple(file, gpuid));
        }

        virtual void worker(promise<bool>& result) override{

            string file = get<0>(start_param_);
            int gpuid   = get<1>(start_param_);

            TRT::set_device(gpuid);
            auto engine = TRT::load_infer(file);
            if(engine == nullptr){
                INFOE("Engine %s load failed", file.c_str());
                result.set_value(false);
                return;
            }

            engine->print();

            TRT::Tensor output_array_device(TRT::DataType::Float);
            int max_batch_size = engine->get_max_batch_size();
            auto input         = engine->tensor("images");
            auto output        = engine->tensor("output");

            input_width_       = input->size(3);
            input_height_      = input->size(2);
            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_            = engine->get_stream();
            gpu_               = gpuid;
            result.set_value(true);

            input->resize_single_dim(0, max_batch_size).to_gpu();

            // resize kernel and D2H copy must be ordered on the engine stream
            output_array_device.set_stream(stream_);

            vector<Job> fetch_jobs;
            while(get_jobs_and_wait(fetch_jobs, max_batch_size)){

                int infer_batch_size = fetch_jobs.size();
                input->resize_single_dim(0, infer_batch_size);

                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job  = fetch_jobs[ibatch];
                    auto& mono = job.mono_tensor->data();

                    if(mono->get_stream() != stream_){
                        // synchronize preprocess stream finish
                        checkCudaRuntime(cudaStreamSynchronize(mono->get_stream()));
                    }

                    input->copy_from_gpu(input->offset(ibatch), mono->gpu(), mono->count());
                    job.mono_tensor->release();
                }

                engine->forward(false);

                int output_width  = output->size(3);
                int output_height = output->size(2);
                int max_width = 0, max_height = 0;
                for(auto& job : fetch_jobs){
                    max_width  = max(max_width,  job.additional.width);
                    max_height = max(max_height, job.additional.height);
                }

                // staging layout: [batch, max_height, max_width], each job uses its own w*h region
                output_array_device.resize(infer_batch_size, max_height, max_width).to_gpu(false);
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job = fetch_jobs[ibatch];
                    resize_depth_align_corners_invoker(
                        output->gpu<float>(ibatch), output_width, output_height,
                        output_array_device.gpu<float>(ibatch),
                        job.additional.width, job.additional.height,
                        stream_
                    );
                }

                output_array_device.to_cpu();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    auto& job = fetch_jobs[ibatch];
                    int width  = job.additional.width;
                    int height = job.additional.height;

                    cv::Mat depth_mat(height, width, CV_32FC1, output_array_device.cpu<float>(ibatch));
                    job.pro->set_value(depth_mat.clone());
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFO("Engine destroy.");
        }

        virtual bool preprocess(Job& job, const Mat& image) override{

            if(tensor_allocator_ == nullptr){
                INFOE("tensor_allocator_ is nullptr");
                return false;
            }

            if(image.empty()){
                INFOE("Image is empty");
                return false;
            }

            if(image.type() != CV_8UC3){
                INFOE("Image type must be CV_8UC3, but got %d", image.type());
                return false;
            }

            job.mono_tensor = tensor_allocator_->query();
            if(job.mono_tensor == nullptr){
                INFOE("Tensor allocator query failed.");
                return false;
            }

            CUDATools::AutoDevice auto_device(gpu_);
            auto& tensor = job.mono_tensor->data();
            TRT::CUStream preprocess_stream = nullptr;

            if(tensor == nullptr){
                // not init
                tensor = make_shared<TRT::Tensor>();
                tensor->set_workspace(make_shared<TRT::MixMemory>());

                if(use_multi_preprocess_stream_){
                    checkCudaRuntime(cudaStreamCreate(&preprocess_stream));

                    // owner = true, stream needs to be free during deconstruction
                    tensor->set_stream(preprocess_stream, true);
                }else{
                    preprocess_stream = stream_;

                    // owner = false, tensor ignored the stream
                    tensor->set_stream(preprocess_stream, false);
                }
            }

            job.additional.width  = image.cols;
            job.additional.height = image.rows;

            cv::Mat image_cont = image.isContinuous() ? image : image.clone();

            preprocess_stream = tensor->get_stream();
            tensor->resize(1, 3, input_height_, input_width_);

            size_t size_image = image_cont.cols * image_cont.rows * 3;
            auto workspace    = tensor->get_workspace();
            uint8_t* gpu_workspace = (uint8_t*)workspace->gpu(size_image);
            uint8_t* cpu_workspace = (uint8_t*)workspace->cpu(size_image);

            memcpy(cpu_workspace, image_cont.data, size_image);
            checkCudaRuntime(cudaMemcpyAsync(gpu_workspace, cpu_workspace, size_image, cudaMemcpyHostToDevice, preprocess_stream));

            // stretch resize to the network input, BGR -> RGB, /255 only
            CUDAKernel::resize_bilinear_and_normalize(
                gpu_workspace,        image_cont.cols * 3,       image_cont.cols,       image_cont.rows,
                tensor->gpu<float>(), input_width_,              input_height_,
                normalize_, preprocess_stream
            );
            return true;
        }

        virtual vector<shared_future<Mat>> commits(const vector<Mat>& images) override{
            return ControllerImpl::commits(images);
        }

        virtual std::shared_future<Mat> commit(const Mat& image) override{
            return ControllerImpl::commit(image);
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int gpu_                    = 0;
        TRT::CUStream stream_       = nullptr;
        bool use_multi_preprocess_stream_ = false;
        CUDAKernel::Norm normalize_;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, int gpuid,
        bool use_multi_preprocess_stream
    ){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(engine_file, gpuid, use_multi_preprocess_stream)){
            instance.reset();
        }
        return instance;
    }

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch){

        CUDAKernel::Norm normalize;
        normalize = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);

        cv::Mat image_cont = image.isContinuous() ? image : image.clone();

        int dst_width  = tensor->size(3);
        int dst_height = tensor->size(2);

        size_t size_image = image_cont.cols * image_cont.rows * 3;
        auto workspace    = tensor->get_workspace();
        uint8_t* gpu_workspace = (uint8_t*)workspace->gpu(size_image);
        uint8_t* cpu_workspace = (uint8_t*)workspace->cpu(size_image);
        auto stream            = tensor->get_stream();

        memcpy(cpu_workspace, image_cont.data, size_image);
        checkCudaRuntime(cudaMemcpyAsync(gpu_workspace, cpu_workspace, size_image, cudaMemcpyHostToDevice, stream));

        CUDAKernel::resize_bilinear_and_normalize(
            gpu_workspace,               image_cont.cols * 3,       image_cont.cols,       image_cont.rows,
            tensor->gpu<float>(ibatch),  dst_width,                dst_height,
            normalize, stream
        );
        tensor->synchronize();
    }
};
