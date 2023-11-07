
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/cuda_tools.hpp>
#include <common/preprocess_kernel.cuh>
#include "app_yolo_cls/yolo_cls.hpp"

using namespace std;

bool requires(const char* name);

static void append_to_file(const string& file, const string& data){
    FILE* f = fopen(file.c_str(), "a+");
    if(f == nullptr){
        INFOE("Open %s failed.", file.c_str());
        return;
    }

    fprintf(f, "%s\n", data.c_str());
    fclose(f);
}

void test_crop_resize(){

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    shared_ptr<TRT::Tensor> tensor = nullptr;
    if(tensor == nullptr){
        tensor = make_shared<TRT::Tensor>();
        tensor->set_workspace(make_shared<TRT::MixMemory>());
        tensor->set_stream(stream, false);
    }

    cv::Mat image = cv::imread("inference/car.jpg");
    
    tensor->resize(1, 3, 224, 224);
    size_t size_image = image.cols * image.rows * 3;
    auto workspace = tensor->get_workspace();
    uint8_t* gpu_workspace = (uint8_t*)workspace->gpu(size_image);
    uint8_t* image_device  = gpu_workspace;

    uint8_t* cpu_workspace = (uint8_t*)workspace->cpu(size_image);
    uint8_t* image_host = cpu_workspace;

    memcpy(image_host, image.data, size_image);
    checkCudaRuntime(cudaMemcpyAsync(image_device, image_host, size_image, cudaMemcpyHostToDevice, stream));        

    CUDAKernel::Norm norm;

    CUDAKernel::crop_resize_bilinear_and_normalize(
        image_device, image.cols * 3, image.cols, image.rows,
        tensor->gpu<float>(), 224, 224, norm, stream 
    );

    cudaStreamSynchronize(stream);
    tensor->to_cpu();
    tensor->save_to_file("crop_resize_cuda.bin");
    INFO("save done.");
    cudaStreamDestroy(stream);
}

static void inference_and_performance(int deviceid, const string& engine_file, TRT::Mode mode, const string& model_name){

    auto engine = YoloCls::create_infer(
        engine_file,                // engine file
        deviceid,                   // gpu id
        false                       // preprocess use multi stream
    );
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    auto files = iLogger::find_files("inference", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    vector<cv::Mat> images;
    for(int i = 0; i < files.size(); ++i){
        auto image = cv::imread(files[i]);
        images.emplace_back(image);
    }

    // warmup
    vector<shared_future<YoloCls::ProbArray>> probs_array;
    for(int i = 0; i < 10; ++i)
        probs_array = engine->commits(images);
    probs_array.back().get();
    probs_array.clear();
    
    /////////////////////////////////////////////////////////
    const int ntest = 100;
    auto begin_timer = iLogger::timestamp_now_float();

    for(int i  = 0; i < ntest; ++i)
        probs_array = engine->commits(images);
    
    // wait all result
    probs_array.back().get();

    float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / ntest / images.size();
    auto mode_name = TRT::mode_string(mode);
    INFO("average time %.2f ms", inference_average_time);
    INFO("%s[YoloV8-Cls] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), inference_average_time, 1000 / inference_average_time);
    append_to_file("perf.result.log", iLogger::format("%s,YoloV8-Cls,%s,%f", model_name.c_str(), mode_name, inference_average_time));

    auto labels = iLogger::split_string(iLogger::load_text_file("imagenet.txt"), "\n");
    for(int i = 0; i < probs_array.size(); ++i){
        auto probs        = probs_array[i].get();
        int predict_label = probs[0].class_label;
        auto predict_name = labels[predict_label];
        float confidence  = probs[0].confidence;
        INFO("%s, The model predict: %s, label = %d, confidence = %.4f", files[i].c_str(), predict_name.c_str(), predict_label, confidence);
    }
    engine.reset();
}

static void test(TRT::Mode mode, const string& model){

    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);
            YoloCls::image_to_tensor(image, tensor, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test YoloV8-Cls %s %s ==================================", mode_name, name);

    if(not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 16;
    
    if(not iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file,                 // save to
            {},
            int8process,
            "inference"
        );
    }

    inference_and_performance(deviceid, model_file, mode, name);
}

static void test_single_image(){
    
    auto engine = YoloCls::create_infer(
        "yolov8s-cls.FP32.trtmodel",            // engine file
        0,                                      // gpu id
        false                                   // preprocess use multi stream
    );
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    cv::Mat image = cv::imread("inference/car.jpg");
    if(image.empty()){
        INFOE("Image is empty");
        return;
    }    

    auto probs        = engine->commit(image).get();
    auto labels       = iLogger::split_string(iLogger::load_text_file("imagenet.txt"), "\n");
    int predict_label = probs[0].class_label;
    auto predict_name = labels[predict_label];
    float confidence  = probs[0].confidence;
    INFO("The model predict: %s, label = %d, confidence = %.4f", predict_name.c_str(), predict_label, confidence);

    engine.reset();
}

int app_yolo_cls(){
 
    test(TRT::Mode::FP32, "yolov8s-cls");
    // test_single_image();
    // test_crop_resize();
    return 0;
}