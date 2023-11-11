#include "rtdetr.hpp"
#include <common/cuda_tools.hpp>
#include <common/infer_controller.hpp>
#include <common/preprocess_kernel.cuh>
#include <common/monopoly_allocator.hpp>

namespace RTDETR{
    using namespace cv;
    using namespace std;

    void decode_kernel_invoker(
        float* predict, int num_bboxes, int num_classes, float confidence_threshold, 
        float* parray, int max_objects, cudaStream_t stream
    );

    using ControllerImpl = InferController
    <
        Mat,                    // input
        BoxArray,               // output
        tuple<string, int>      // start param
    >;
    class InferImpl : public Infer, public ControllerImpl{
    public:

        /** 要求在InferImpl里面执行stop，而不是在基类执行stop **/
        virtual ~InferImpl(){
            stop();
        }        

        virtual bool startup(
            const string& file, int gpuid, float confidence_threshold,
            int max_objects, bool use_multi_preprocess_stream
        ){
            normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);            
            max_objects_                 = max_objects;
            confidence_threshold_        = confidence_threshold;
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

            const int MAX_IMAGE_BBOX  = max_objects_;
            const int NUM_BOX_ELEMENT = 6;      // left, top, right, bottom, confidence, class
            TRT::Tensor output_array_device(TRT::DataType::Float);
            int max_batch_size = engine->get_max_batch_size();
            auto input         = engine->tensor("images");
            auto output        = engine->tensor("output");
            int num_classes    = output->size(2) - 4;

            input_width_       = input->size(3);
            input_height_      = input->size(2);
            tensor_allocator_  = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_            = engine->get_stream();
            gpu_               = gpuid;
            result.set_value(true); 

            input->resize_single_dim(0, max_batch_size).to_gpu();

            // 这里的 1 + MAX_IMAGE_BBOX结构是，counter + bboxes ...
            output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_BBOX * NUM_BOX_ELEMENT).to_gpu();                                   
        
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
                output_array_device.to_gpu(false);
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    
                    auto& job                 = fetch_jobs[ibatch];
                    float* image_based_output = output->gpu<float>(ibatch);
                    float* output_array_ptr   = output_array_device.gpu<float>(ibatch);
                    checkCudaRuntime(cudaMemsetAsync(output_array_ptr, 0, sizeof(int), stream_));
                    decode_kernel_invoker(image_based_output, output->size(1), num_classes, confidence_threshold_, output_array_ptr, MAX_IMAGE_BBOX, stream_);
                }

                output_array_device.to_cpu();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    float* parray = output_array_device.cpu<float>(ibatch);
                    int count     = min(MAX_IMAGE_BBOX, (int)*parray);
                    auto& job     = fetch_jobs[ibatch];
                    auto& image_based_boxes   = job.output;
                    for(int i = 0; i < count; ++i){
                        float* pbox  = parray + 1 + i * NUM_BOX_ELEMENT;
                        int label    = pbox[5];
                        image_based_boxes.emplace_back(pbox[0], pbox[1], pbox[2], pbox[3], pbox[4], label);
                    }
                    job.pro->set_value(image_based_boxes);
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

            Size input_size(input_width_, input_height_);
            
            preprocess_stream = tensor->get_stream();
            tensor->resize(1, 3, input_height_, input_width_);

            size_t size_image      = image.cols * image.rows * 3;
            auto workspace         = tensor->get_workspace();
            uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_image);
            float*   affine_matrix_device = (float*)gpu_workspace;
            uint8_t* image_device         = gpu_workspace;

            uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_image);
            float* affine_matrix_host     = (float*)cpu_workspace;
            uint8_t* image_host           = cpu_workspace;

            //checkCudaRuntime(cudaMemcpyAsync(image_host,   image.data, size_image, cudaMemcpyHostToHost,   stream_));
            // speed up
            memcpy(image_host, image.data, size_image);
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, preprocess_stream));

            CUDAKernel::resize_bilinear_and_normalize(
                image_device,         image.cols * 3,  image.cols,     image.rows, 
                tensor->gpu<float>(), input_width_,    input_height_,  normalize_, 
                preprocess_stream
            );
            return true;
        }        

        virtual vector<shared_future<BoxArray>> commits(const vector<Mat>& images) override{
            return ControllerImpl::commits(images);
        }

        virtual std::shared_future<BoxArray> commit(const Mat& image) override{
            return ControllerImpl::commit(image);
        }
    
    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int gpu_                    = 0;
        float confidence_threshold_ = 0;
        int max_objects_            = 300;
        TRT::CUStream stream_       = nullptr;
        bool use_multi_preprocess_stream_ = false;
        CUDAKernel::Norm normalize_;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, int gpuid, float confidence_threshold,
        int max_object, bool use_multi_preprocess_stream
    ){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(
            engine_file, gpuid, confidence_threshold,
            max_object, use_multi_preprocess_stream)
        ){
            instance.reset();            
        }
        return instance;
    }

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch){

        CUDAKernel::Norm normalize;
        normalize = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::Invert);
        Size input_size(tensor->size(3), tensor->size(2));

        size_t size_image      = image.cols * image.rows * 3;
        auto workspace         = tensor->get_workspace();
        uint8_t* gpu_workspace        = (uint8_t*)workspace->gpu(size_image);
        uint8_t* image_device         = gpu_workspace;

        uint8_t* cpu_workspace        = (uint8_t*)workspace->cpu(size_image);
        uint8_t* image_host           = cpu_workspace;
        auto stream                   = tensor->get_stream();

        memcpy(image_host, image.data, size_image);
        checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));

        CUDAKernel::resize_bilinear_and_normalize(
            image_device,               image.cols * 3,       image.cols,         image.rows, 
            tensor->gpu<float>(ibatch), input_size.width,     input_size.height,  normalize,  stream
        );
        tensor->synchronize();
    }    
};