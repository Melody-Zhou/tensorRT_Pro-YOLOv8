#include "ppocr_rec.hpp"
#include <fstream>
#include <common/infer_controller.hpp>
#include <common/preprocess_kernel.cuh>

namespace PaddleOCR{
namespace SVTRRecognizer{

    using namespace cv;
    using namespace std;

    template <class ForwardIterator>
    inline static size_t argmax(ForwardIterator first, ForwardIterator last) {
        return std::distance(first, std::max_element(first, last));
    }
    static bool read_dict(const string& path, vector<string>& label_list){
        if(iLogger::exists(path)){
            std::ifstream in(path);
            std::string line;
            while(getline(in, line)){
                label_list.push_back(line);
            }
            return true;
        }else{
            INFOE("no such label file: %s", path.c_str());
            return false;
        }
    }

    using ControllerImpl = InferController
    <
        Mat,                    // input
        Text,                   // output
        tuple<string, int>      // start param    
    >;
    class TextRecognizerImpl : public TextRecognizer, public ControllerImpl{
    public:
        
        /** 要求在TextRecognizerImpl里面执行stop，而不是在基类执行stop **/
        virtual ~TextRecognizerImpl(){
            stop();
        }        

        virtual bool startup(
            const string& engine_file, int gpuid, const string& dict_path,
            int rec_batch_num, bool use_multi_preprocess_stream
        ){
            float mean[3] = {0.5, 0.5, 0.5};
            float std[3]  = {0.5, 0.5, 0.5};
            normalize_ = CUDAKernel::Norm::mean_std(mean, std, 1 / 255.0, CUDAKernel::ChannelType::None);

            if(!(read_dict(dict_path, label_list_))){
                INFOE("read dict failed");
                return false;
            }
            // blank char for ctc
            label_list_.insert(label_list_.begin(), "#");
            label_list_.push_back(" ");

            use_multi_preprocess_stream_ = use_multi_preprocess_stream;
            rec_batch_num_ = rec_batch_num;
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

            INFO("Recognizer engine info: ");
            INFO("============================================");
            engine->print();
            INFO("============================================");
            TRT::Tensor output_array_device(TRT::DataType::Float);
            // int max_batch_size = engine->get_max_batch_size();
            int max_batch_size = std::min(engine->get_max_batch_size(), rec_batch_num_);
            auto input  = engine->tensor("images");
            auto output = engine->tensor("output");

            input_width_      = input->size(3);
            input_height_     = input->size(2);
            tensor_allocator_ = make_shared<MonopolyAllocator<TRT::Tensor>>(max_batch_size * 2);
            stream_           = engine->get_stream();
            gpu_              = gpuid;
            result.set_value(true);

            input->resize_single_dim(0, max_batch_size).to_gpu();

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
                int max_chars  = output->size(1);
                int vocab_size = output->size(2);
                for(int ibatch = 0; ibatch < infer_batch_size; ++ibatch){
                    
                    float* parray = output->cpu<float>(ibatch);
                    auto& job     = fetch_jobs[ibatch];
                    auto& image_based_text = job.output;

                    // batch, max_chars, vocab_size
                    // parry->1x80x6625
                    int argmax_idx;
                    int count       = 0;
                    int last_index  = 0;
                    float score     = 0.0f;
                    float max_value = 0.0f;
                    
                    for(int i = 0; i < max_chars; ++i){
                        // get idx
                        argmax_idx = int(argmax(&parray[i * vocab_size], &parray[(i + 1) * vocab_size]));
                        // get score
                        max_value  = float(*std::max_element(&parray[i * vocab_size], &parray[(i + 1) * vocab_size]));
                        
                        if(argmax_idx > 0 && (!(i > 0 && argmax_idx == last_index))){
                            score   += max_value;
                            count   += 1;
                            image_based_text.text += label_list_[argmax_idx];
                        }
                        last_index = argmax_idx;
                    }
                    if(count != 0){
                        score /= count;
                    }
                    image_based_text.score = score;
                    job.pro->set_value(image_based_text);
                }
                fetch_jobs.clear();
            }
            stream_ = nullptr;
            tensor_allocator_.reset();
            INFO("Recognizer engine destroy");
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
        
            float ratio = (float)(image.cols) / (image.rows);
            int resized_w;
            if(std::ceil(input_height_ * ratio) > input_width_){
                resized_w = input_width_;
            }else{
                resized_w = (int)(std::ceil(input_height_ * ratio));
            }

            CUDAKernel::resize_normalize_image(
                image_device,         image.cols * 3,   image.cols,     image.rows,
                tensor->gpu<float>(), input_width_,     input_height_,  resized_w,
                normalize_,           preprocess_stream
            );

            return true;
        }

        virtual shared_future<Text> commit(const Mat& image) override{
            return ControllerImpl::commit(image);
        }

        virtual vector<shared_future<Text>> commits(const vector<Mat>& images) override{
            return ControllerImpl::commits(images);
        }

    private:
        int gpu_              = 0;
        int input_width_      = 0;
        int input_height_     = 0;
        int rec_batch_num_    = 0;
        TRT::CUStream stream_ = nullptr;
        bool use_multi_preprocess_stream_ = false;
        vector<string> label_list_;
        CUDAKernel::Norm normalize_;
    };
    
    shared_ptr<TextRecognizer> create_recognizer(
        const string& engine_file, int gpuid, const string& dict_path,
        int rec_batch_num, bool use_multi_preprocess_stream
    ){
        shared_ptr<TextRecognizerImpl> instance(new TextRecognizerImpl());
        if(!instance->startup(
            engine_file, gpuid, dict_path, rec_batch_num,
            use_multi_preprocess_stream)
        ){
            instance.reset();
        }
        return instance;
    }

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch){

        CUDAKernel::Norm normalize;
        float mean[3] = {0.5, 0.5, 0.5};
        float std[3]  = {0.5, 0.5, 0.5};
        normalize = CUDAKernel::Norm::mean_std(mean, std, 1 / 255.0, CUDAKernel::ChannelType::None);        
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

        float ratio = (float)(image.cols) / (image.rows);
        int resized_w;
        if(std::ceil(input_size.height * ratio) > input_size.width){
            resized_w = input_size.width;
        }else{
            resized_w = (int)(std::ceil(input_size.height * ratio));
        }

        CUDAKernel::resize_normalize_image(
            image_device,         image.cols * 3,   image.cols,        image.rows,
            tensor->gpu<float>(), input_size.width, input_size.height, resized_w,
            normalize,            stream
        );        
    }
};
};