#include "depth_anything.hpp"
#include <common/infer_controller.hpp>
#include <common/preprocess_kernel.cuh>

namespace DepthAnything{

    using namespace cv;
    using namespace std;

    void resize_depth_image(
        float* src_depth, int src_width, int src_height,
        float* dst_depth, int dst_width, int dst_height,
        cudaStream_t stream
    );

    const char* type_name(Type type){
        switch(type){
        case Type::V1: return "Depth-Anything-V1";
        case Type::V2: return "Depth-Anything-V2";
        default: return "Unknow";
        }
    }

    using ControllerImpl = InferController
    <
        Mat,                    // input
        Mat,                    // output
        tuple<string, int>      // start param
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:

        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            stop();
        }

        virtual bool startup(
            const string& file, int gpuid,
            InterpolationDevice interpolation_device,
            bool use_multi_preprocess_stream
        ){
            float mean[3] = {0.485, 0.456, 0.406};
            float std[3]  = {0.229, 0.224, 0.225};
            normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1 / 255.0f, CUDAKernel::ChannelType::Invert);
            interpolation_device_ = interpolation_device;
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

            if(interpolation_device_ == InterpolationDevice::FastGPU){
                output_array_device.resize(max_batch_size, max_input_width_ * max_input_height_).to_gpu();
            }

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

                auto t1 = iLogger::timestamp_now_float();
                if(interpolation_device_ == InterpolationDevice::FastGPU){
                    for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                        float* parry = output->gpu<float>(ibatch);
                        float* output_array_ptr = output_array_device.gpu<float>(ibatch);
                        resize_depth_image(parry, input_width_, input_height_, output_array_ptr, src_width_, src_height_, stream_);
                    }

                    output_array_device.to_cpu();
                    for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                        auto& job         = fetch_jobs[ibatch];
                        auto& depth_image = job.output;
                        float* parry  = output_array_device.cpu<float>(ibatch);

                        cv::Mat depth_mat(src_height_, src_width_, CV_32FC1, parry);
                        depth_image = depth_mat.clone();
                        job.pro->set_value(depth_image);
                    }
                    fetch_jobs.clear();
                }
                
                if(interpolation_device_ == InterpolationDevice::CPU){
                    for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                        auto& job         = fetch_jobs[ibatch];
                        auto& depth_image = job.output;
                        float* parry  = output->cpu<float>(ibatch);

                        cv::Mat depth_mat(input_height_, input_width_, CV_32FC1, parry);
                        depth_image = depth_mat.clone();
                        job.pro->set_value(depth_image);
                    }
                    fetch_jobs.clear();
                }
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

            src_width_  = image.cols;
            src_height_ = image.rows;
            Size input_size(input_width_, input_height_);
            preprocess_stream = tensor->get_stream();
            tensor->resize(1, 3, input_height_, input_width_);

            size_t size_image = image.cols * image.rows * 3;
            auto workspace    = tensor->get_workspace();
            uint8_t* gpu_workspace = (uint8_t*)workspace->gpu(size_image);
            uint8_t* image_device  = gpu_workspace;

            uint8_t* cpu_workspace = (uint8_t*)workspace->cpu(size_image);
            uint8_t* image_host    = cpu_workspace;

            // speed up
            memcpy(image_host, image.data, size_image);
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, preprocess_stream));

            CUDAKernel::resize_bilinear_and_normalize(
                image_device,         image.cols * 3,   image.cols,     image.rows,
                tensor->gpu<float>(), input_width_,     input_height_,
                normalize_,           preprocess_stream
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
        int gpu_              = 0;
        int src_width_        = 0;
        int src_height_       = 0;
        int input_width_      = 0;
        int input_height_     = 0;
        int max_input_width_  = 4096; // 4K
        int max_input_height_ = 2160; // 4K
        TRT::CUStream stream_ = nullptr;
        InterpolationDevice interpolation_device_ = InterpolationDevice::CPU; 
        bool use_multi_preprocess_stream_ = false;
        CUDAKernel::Norm normalize_;
    };    

    shared_ptr<Infer> create_infer(
        const string& engine_file, Type type, int gpuid,
        InterpolationDevice interpolation_device, bool use_multi_preprocess_stream
    ){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(engine_file, gpuid, interpolation_device, use_multi_preprocess_stream)
        ){
            instance.reset();
        }
        return instance;
    }

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch){
        
        CUDAKernel::Norm normalize;
        float mean[3] = {0.485, 0.456, 0.406};
        float std[3]  = {0.229, 0.224, 0.225};
        normalize = CUDAKernel::Norm::mean_std(mean, std, 1 / 255.0f, CUDAKernel::ChannelType::Invert);
        Size input_size(tensor->size(3), tensor->size(2));

        size_t size_image      = image.cols * image.rows * 3;
        auto workspace         = tensor->get_workspace();
        uint8_t* gpu_workspace = (uint8_t*)workspace->gpu(size_image);
        uint8_t* image_device  = gpu_workspace;
        
        uint8_t* cpu_workspace = (uint8_t*)workspace->cpu(size_image);
        uint8_t* image_host    = cpu_workspace;
        auto stream            = tensor->get_stream();
        
        memcpy(image_host, image.data, size_image);
        checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));

        CUDAKernel::resize_bilinear_and_normalize(
            image_device,               image.cols * 3,   image.cols,        image.rows,
            tensor->gpu<float>(ibatch), input_size.width, input_size.height,
            normalize, stream
        );        
        tensor->synchronize();
    }
};