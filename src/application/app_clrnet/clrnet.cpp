#include "clrnet.hpp"
#include <common/ilogger.hpp>
#include <common/infer_controller.hpp>
#include <common/preprocess_kernel.cuh>
#include <common/monopoly_allocator.hpp>
#include <common/cuda_tools.hpp>

namespace CLRNet{

    using namespace cv;
    using namespace std;

    void decode_kernel_invoker(
        float* predict, int num_lanes, float confidence_threshold,
        float* parray, cudaStream_t stream
    );

    void nms_kernel_invoker(
        float* parray, float nms_threshold, int max_lanes, 
        int input_width, cudaStream_t stream
    );

    static float LaneIoU(const Lane& a, const Lane& b, int input_width){
        int start_a = (int)(a.start_y * N_STRIPS + 0.5f);
        int start_b = (int)(b.start_y * N_STRIPS + 0.5f);
        int start   = std::max(start_a, start_b);
        int end_a   = start_a + (int)(a.length + 0.5f) - 1;
        int end_b   = start_b + (int)(b.length + 0.5f) - 1;
        int end     = std::min(std::min(end_a, end_b), N_STRIPS);
        float dist  = 0.0f;
        for(int i = start; i <= end; ++i){
            dist += fabs(a.lane_xs[i] - b.lane_xs[i]);
        }
        dist = dist * (input_width - 1) / (float)(end - start + 1);
        return dist;
    }

    static LaneArray cpu_nms(LaneArray& lanes, float threshold, int nms_topk, int input_width){

        LaneArray output;
        output.reserve(lanes.size());

        vector<bool> remove_flags(lanes.size());
        for(int i = 0; i < lanes.size(); ++i){
            if(remove_flags[i]) continue;

            auto& a = lanes[i];
            output.emplace_back(a);
            if(output.size() == nms_topk){
                break;
            }

            for(int j = i + 1; j < lanes.size(); ++j){
                if(remove_flags[j]) continue;

                auto& b = lanes[j];
                if(LaneIoU(a, b, input_width) < threshold)
                    remove_flags[j] = true;
            }
        }
        return output;
    }

    using ControllerImpl = InferController
    <
        Mat,                    // input
        LaneArray,              // output
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
            float nms_threshold, int nms_topk, int cut_height, NMSMethod nms_method, 
            int max_lanes, bool use_multi_preprocess_stream 
        ){
            normalize_ = CUDAKernel::Norm::alpha_beta(1 / 255.0f, 0.0f, CUDAKernel::ChannelType::None);

            use_multi_preprocess_stream_ = use_multi_preprocess_stream,
            confidence_threshold_ = confidence_threshold;
            nms_threshold_        = nms_threshold;
            nms_topk_             = nms_topk;
            cut_height_           = cut_height;
            max_lanes_            = max_lanes;
            nms_method_           = nms_method;
            anchor_ys_.reserve(N_OFFSETS);
            for(int i = 0; i < N_OFFSETS; ++i){
                auto step = 1.0f - i / float(N_STRIPS);
                anchor_ys_[i] = step;
            }
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

            input_width_      = input->size(3);
            input_height_     = input->size(2);
            tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_           = engine->get_stream();
            gpu_              = gpuid;
            result.set_value(true);

            input->resize_single_dim(0, max_batch_size).to_gpu();

            // 这里的 1 + MAX_IMAGE_LANE 结构是 counter + lanes ...
            const int MAX_IMAGE_LANE   = output->size(1);
            const int NUM_LANE_ELEMENT = output->size(2);
            output_array_device.resize(max_batch_size, 1 + MAX_IMAGE_LANE * NUM_LANE_ELEMENT).to_gpu();

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
                    decode_kernel_invoker(image_based_output, output->size(1), confidence_threshold_, output_array_ptr, stream_);
                    
                    if(nms_method_ == NMSMethod::FastGPU){
                        nms_kernel_invoker(output_array_ptr, nms_threshold_, max_lanes_, input_width_, stream_);
                    }
                }

                output_array_device.to_cpu();
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    float* parray = output_array_device.cpu<float>(ibatch);
                    int count = min((int)*parray, max_lanes_);
                    auto& job = fetch_jobs[ibatch];
                    auto& image_based_lanes = job.output;
                    image_based_lanes.reserve(count);

                    for(int i = 0; i < count; ++i){
                        float* plane = parray + 1 + i * (NUM_LANE_ELEMENT + 1);
                        int keepflag = plane[6];
                        if(keepflag == 0)
                            continue;

                        Lane lane;
                        lane.unknow  = plane[0];
                        lane.score   = plane[1];
                        lane.start_y = plane[2];
                        lane.start_x = plane[3];
                        lane.theta   = plane[4];
                        lane.length  = plane[5];
                        for(int i = 0; i < N_OFFSETS; ++i){
                            lane.lane_xs[i] = plane[i + 7];
                        }
                        image_based_lanes.push_back(lane);
                    }
                    // sort
                    std::sort(image_based_lanes.begin(), image_based_lanes.end(), [](LaneArray::const_reference a, LaneArray::const_reference b){
                        return a.score > b.score;
                    });
                    if(nms_method_ == NMSMethod::CPU){
                        image_based_lanes = cpu_nms(image_based_lanes, nms_threshold_, nms_topk_, input_width_);
                    }else if(nms_method_ == NMSMethod::FastGPU){
                        if(image_based_lanes.size() > nms_topk_){
                            image_based_lanes.resize(nms_topk_);
                        }
                    }
                    for(auto& lane : image_based_lanes){
                        lane.points.reserve(N_OFFSETS / 2);
                        int start = (int)(lane.start_y * N_STRIPS + 0.5f);
                        int end   = start + (int)(lane.length + 0.5f) - 1;
                        end       = min(end, N_STRIPS);
                        for(int i = start; i <= end; ++i){
                            lane.points.push_back(cv::Point2f(lane.lane_xs[i], anchor_ys_[i]));
                        }
                    }
                    job.pro->set_value(image_based_lanes);                                        
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

            if(cut_height_ >= image.cols){
                INFOE("cut_height is too large: %d, src_height: %d", cut_height_, image.cols);
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
            auto workspapce        = tensor->get_workspace();
            uint8_t* gpu_workspace = (uint8_t*)workspapce->gpu(size_image);
            uint8_t* image_device  = gpu_workspace;

            uint8_t* cpu_workspace = (uint8_t*)workspapce->cpu(size_image);
            uint8_t* image_host    = cpu_workspace;

            // speed up
            memcpy(image_host, image.data, size_image);
            checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, preprocess_stream));            
            CUDAKernel::cut_resize_bilinear_and_normalize(
                image_device,          image.cols * 3,   image.cols,     image.rows,
                tensor->gpu<float>(),  input_width_,     input_height_,  cut_height_,
                normalize_,            preprocess_stream
            );
            return true;
        }

        virtual shared_future<LaneArray> commit(const Mat& image) override{
            return ControllerImpl::commit(image);
        }

        virtual vector<shared_future<LaneArray>> commits(const vector<Mat>& images) override{
            return ControllerImpl::commits(images); 
        }

    private:
        int input_width_            = 0;
        int input_height_           = 0;
        int cut_height_             = 0;
        int gpu_                    = 0;
        float confidence_threshold_ = 0;
        float nms_threshold_        = 0;
        float nms_topk_             = 0;     
        int max_lanes_              = 0;
        NMSMethod nms_method_       = NMSMethod::FastGPU;
        TRT::CUStream stream_       = nullptr;
        bool use_multi_preprocess_stream_ = false;
        CUDAKernel::Norm normalize_;
        vector<float> anchor_ys_;  

    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, int gpuid, float confidence_threshold, 
        float nms_threshold, int nms_topk, int cut_height, NMSMethod nms_method, 
        int max_lanes, bool use_multi_preprocess_stream
    ){
        shared_ptr<InferImpl> instance(new InferImpl());
        if(!instance->startup(
            engine_file, gpuid, confidence_threshold, nms_threshold,
            nms_topk, cut_height, nms_method, max_lanes, use_multi_preprocess_stream)
        ){
            instance.reset();
        }
        return instance;
    }

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch, int cut_height){

        auto normalize = CUDAKernel::Norm::alpha_beta(1/ 255.0f, 0.0f, CUDAKernel::ChannelType::None);
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
        
        CUDAKernel::cut_resize_bilinear_and_normalize(
            image_device,                image.cols * 3,    image.cols,         image.rows,
            tensor->gpu<float>(ibatch),  input_size.width,  input_size.height,  cut_height,
            normalize,                   stream
        );
        tensor->synchronize();
    }
};