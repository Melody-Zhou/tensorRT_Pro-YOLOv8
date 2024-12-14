#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_depth_anything/depth_anything.hpp"
#include "tools/zmq_remote_show.hpp"

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

static void inference_and_performance(
    int deviceid, const string& engine_file, TRT::Mode mode, DepthAnything::Type type, 
    DepthAnything::InterpolationDevice interpolation_device, const string& model_name
){

    auto engine = DepthAnything::create_infer(
        engine_file,                                // engine file
        type,                                       // depth anything type
        deviceid,                                   // gpu id
        interpolation_device,                       // cpu/fastgpu
        false                                       // preprocess use multi stream
    );

    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    auto files = iLogger::find_files("inference_depth/images", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    vector<cv::Mat> images;
    for(int i = 0; i < files.size(); ++i){
        auto image = cv::imread(files[i]);
        images.emplace_back(image);
    }    

    // warmup
    vector<shared_future<cv::Mat>> depth_image_array;
    for(int i = 0; i < 10; ++i)
        depth_image_array = engine->commits(images);
    depth_image_array.back().get();
    depth_image_array.clear();

    /////////////////////////////////////////////////////////
    const int ntest = 100;
    auto begin_timer = iLogger::timestamp_now_float();

    for(int i  = 0; i < ntest; ++i)
        depth_image_array = engine->commits(images);
    
    // wait all result
    depth_image_array.back().get();

    float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / ntest / images.size();
    auto type_name = DepthAnything::type_name(type);
    auto mode_name = TRT::mode_string(mode);
    INFO("%s[%s] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), type_name, inference_average_time, 1000 / inference_average_time);
    append_to_file("perf.result.log", iLogger::format("%s,%s,%s,%f", model_name.c_str(), type_name, mode_name, inference_average_time));

    string root = iLogger::format("%s_%s_%s_result", model_name.c_str(), type_name, mode_name);
    iLogger::rmtree(root);
    iLogger::mkdir(root);

    cv::Mat result;
    for(int i = 0; i < depth_image_array.size(); ++i){
        auto depth_image = depth_image_array[i].get();
        if(interpolation_device == DepthAnything::InterpolationDevice::CPU){
            cv::resize(depth_image, depth_image, cv::Size(images[i].cols, images[i].rows));
        }

        // visualization
        cv::normalize(depth_image, depth_image, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(depth_image, depth_image, cv::COLORMAP_INFERNO);
        string file_name = iLogger::file_name(files[i], false);
        string save_path = iLogger::format("%s/%s.jpg", root.c_str(), file_name.c_str());
        INFO("Save to %s, average time %.2f ms", save_path.c_str(), inference_average_time);
        cv::imwrite(save_path, depth_image);
        // cv::hconcat(images[i], depth_image, result);
        // cv::imwrite(save_path, result);
    }
    engine.reset();
}

static void test(DepthAnything::Type type, TRT::Mode mode, DepthAnything::InterpolationDevice interpolation_device, const string& model){

    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);
            DepthAnything::image_to_tensor(image, tensor, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test %s %s %s ==================================", DepthAnything::type_name(type), mode_name, name);        

    if(not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 1;    // be careful
    
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

    inference_and_performance(deviceid, model_file, mode, type, interpolation_device, name);
}

static void test_video(DepthAnything::InterpolationDevice interpolation_device){

    auto engine = DepthAnything::create_infer(
        "depth_anything_v2_vits.sim.FP16.trtmodel",     // engine file
        DepthAnything::Type::V2,                        // depth anything type
        0,                                              // gpu id
        interpolation_device,                           // cpu/fastgpu
        false                                           // preprocess use multi stream
    );

    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    cv::Mat frame;
    cv::VideoCapture cap("inference_depth/videos/davis_dolphins.mp4");
    // cv::VideoCapture cap(0);    // usb camera
    if(!cap.isOpened()){
        INFOE("Camera open failed");
        return;
    }    

    int width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);

    cv::VideoWriter output_video("result_depth_anything.mp4", cv::VideoWriter::fourcc('M', 'J', 'P', 'G'), fps, cv::Size(width * 2, height));

    // auto remote_show = create_zmq_remote_show();
    cv::Mat result;
    while(cap.read(frame)){

        auto t0 = iLogger::timestamp_now_float();
        auto depth_image = engine->commit(frame).get();
        if(interpolation_device == DepthAnything::InterpolationDevice::CPU){
            cv::resize(depth_image, depth_image, cv::Size(frame.cols, frame.rows));
        }
        auto fee = iLogger::timestamp_now_float() - t0;
        INFO("fee = %.2f ms, FPS = %.2f", fee, 1 / fee * 1000);

        // visualization
        cv::normalize(depth_image, depth_image, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::applyColorMap(depth_image, depth_image, cv::COLORMAP_INFERNO);
        cv::hconcat(frame, depth_image, result);
        output_video.write(result);
        // remote_show->post(frame);
        cv::imshow("Depth: Before -> After", result);
        int key = cv::waitKey(1);
        if(key == 27)
            break;
    }
    
    // char end_signal = 'x';
    // remote_show->post(&end_signal, 1);
    INFO("Done.");
    cap.release();
    cv::destroyAllWindows();
    engine.reset();
}

static void test_single_image(DepthAnything::InterpolationDevice interpolation_device){
    
    auto engine = DepthAnything::create_infer(
        "depth_anything_v2_vits.sim.FP16.trtmodel",     // engine file
        DepthAnything::Type::V2,                        // depth anything type
        0,                                              // gpu id
        interpolation_device,                           // cpu/fastgpu
        false                                           // preprocess use multi stream
    );

    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    cv::Mat image = cv::imread("inference_depth/images/demo01.jpg");
    if(image.empty()){
        INFOE("Image is empty");
        return;
    }

    auto depth_image = engine->commit(image).get();
    if(interpolation_device == DepthAnything::InterpolationDevice::CPU){
        cv::resize(depth_image, depth_image, cv::Size(image.cols, image.rows));
    }

    // visualization
    cv::normalize(depth_image, depth_image, 0, 255, cv::NORM_MINMAX, CV_8U);
    cv::applyColorMap(depth_image, depth_image, cv::COLORMAP_INFERNO);
    // cv::imwrite("Result-Depth-Anything.jpg", depth_image);
    cv::Mat result;
    cv::hconcat(image, depth_image, result);
    cv::imwrite("Result-Depth-Anything.jpg", result);
    INFO("Save to Result-Depth-Anything.jpg");
    engine.reset();
}

int app_depth_anything(){

    test(DepthAnything::Type::V2, TRT::Mode::FP16, DepthAnything::InterpolationDevice::CPU, "depth_anything_v2_vits.sim");
    // test_video(DepthAnything::InterpolationDevice::CPU);
    // test_single_image(DepthAnything::InterpolationDevice::CPU);
}