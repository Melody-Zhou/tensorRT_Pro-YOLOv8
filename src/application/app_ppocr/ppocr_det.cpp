#include "ppocr_det.hpp"
#include <common/infer_controller.hpp>
#include <common/preprocess_kernel.cuh>

namespace PaddleOCR{
namespace DBDetector{

    using namespace cv;
    using namespace std;

    void detector_postprocess(
        const Mat& pred_map, BoxArray& boxes, int src_h, int src_w, int dst_h, int dst_w,
        float mask_thresh, float box_thresh, float unclip_ratio, int min_size, int max_candidates
    );

    using ControllerImpl = InferController
    <
        Mat,                    // input
        BoxArray,               // output
        tuple<string, int>      // start param
    >;
    class TextDetectorImpl : public TextDetector, public ControllerImpl{
    public:

        /** 要求在TextDetectorImpl里面执行stop，而不是在基类执行stop **/
        virtual ~TextDetectorImpl(){
            stop();
        }

        virtual bool startup(
            const string& engine_file, int gpuid,
            float mask_thresh, float box_thresh,
            float unclip_ratio, int min_size, int max_candidates,
            bool use_multi_preprocess_stream
        ){
            float mean[3] = {0.485, 0.456, 0.406};
            float std[3]  = {0.229, 0.224, 0.225};
            normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1 / 255.0f, CUDAKernel::ChannelType::None);
            
            use_multi_preprocess_stream_ = use_multi_preprocess_stream;
            mask_thresh_    = mask_thresh;
            box_thresh_     = box_thresh;
            unclip_ratio_   = unclip_ratio;
            min_size_       = min_size;
            max_candidates_ = max_candidates;
            return ControllerImpl::startup(make_tuple(engine_file, gpuid));
        }

        virtual void worker(promise<bool>& result) override{

            string engine_file = get<0>(start_param_);
            int gpuid          = get<1>(start_param_);
            auto engine = TRT::load_infer(engine_file);
            if(engine == nullptr){
                INFOE("Engine %s load failed", engine_file.c_str());
                result.set_value(false);
                return;
            }

            INFO("Detector engine info: ");
            INFO("============================================");
            engine->print();
            INFO("============================================");
            TRT::Tensor output_array_device(TRT::DataType::Float);
            int max_batch_size = engine->get_max_batch_size();
            auto input  = engine->tensor("images");
            auto output = engine->tensor("output");

            input_width_      = input->size(3);
            input_height_     = input->size(2);
            tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_           = engine->get_stream();
            gpu_              = gpuid;
            result.set_value(true);
            
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

                    float* parry = output->cpu<float>(ibatch);
                    auto& job = fetch_jobs[ibatch];
                    auto& image_baesd_boxes = job.output;
                    
                    Mat pred_map(input_height_, input_width_, CV_32FC1, parry);
                    detector_postprocess(
                        pred_map, image_baesd_boxes, src_height_, src_width_, input_height_, input_width_, 
                        mask_thresh_, box_thresh_, unclip_ratio_, min_size_, max_candidates_
                    );

                    job.pro->set_value(image_baesd_boxes);
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFO("Detector engine destroy");
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
                INFOE("Tensor allocator query failed");
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

                    // owner = true, stream needs to free during deconstruction
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

            size_t size_image      = image.cols * image.rows * 3;
            auto workspace         = tensor->get_workspace();
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

        virtual shared_future<BoxArray> commit(const Mat& image) override{
            return ControllerImpl::commit(image);
        }        

    private:
        int src_width_        = 0;
        int src_height_       = 0;
        int input_width_      = 0;
        int input_height_     = 0;
        int gpu_              = 0;
        float mask_thresh_    = 0;
        float box_thresh_     = 0;
        float unclip_ratio_   = 0;
        int min_size_         = 0;
        int max_candidates_   = 1000;
        TRT::CUStream stream_ = nullptr;
        bool use_multi_preprocess_stream_ = false;
        CUDAKernel::Norm normalize_;
    };    

    shared_ptr<TextDetector> create_detector(
        const string& engine_file, int gpuid,
        float mask_thresh, float box_thresh,
        float unclip_ratio, int min_size, int max_candidates,
        bool use_multi_preprocess_stream
    ){
        shared_ptr<TextDetectorImpl> instance(new TextDetectorImpl());
        if(!instance->startup(
            engine_file, gpuid, mask_thresh, box_thresh, unclip_ratio,
            min_size, max_candidates, use_multi_preprocess_stream)
        ){
            instance.reset();
        }
        return instance;
    }

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch){
        
        CUDAKernel::Norm normalize;
        float mean[3] = {0.485, 0.456, 0.406};
        float std[3]  = {0.229, 0.224, 0.225};
        normalize = CUDAKernel::Norm::mean_std(mean, std, 1 / 255.0f, CUDAKernel::ChannelType::None);
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
};