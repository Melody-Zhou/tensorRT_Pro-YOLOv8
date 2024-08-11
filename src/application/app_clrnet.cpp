#include "app_clrnet/clrnet.hpp"
#include <common/ilogger.hpp>
#include <builder/trt_builder.hpp>
#include <tools/zmq_remote_show.hpp>

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

static void inference_and_performance(int deviceid, const string& engine_file, TRT::Mode mode, const string& model_name){

    auto engine = CLRNet::create_infer(
        engine_file,                            // engine file
        deviceid,                               // gpu id
        0.4f,                                   // confidence threshold
        50.0f,                                  // nms threshold
        4,                                      // nms topk
        270,                                    // cut height
        CLRNet::NMSMethod::FastGPU,             // NMS method, fast GPU / CPU
        128,                                    // max lanes
        false                                   // preprocess use multi stream
    );  
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    auto files = iLogger::find_files("inference_lane/images", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    vector<cv::Mat> images;
    for(int i = 0; i < files.size(); ++i){
        auto image = cv::imread(files[i]);
        images.emplace_back(image);
    }

    // warmup
    vector<shared_future<CLRNet::LaneArray>> lanes_array;
    for(int i = 0; i < 10; ++i)
        lanes_array = engine->commits(images);
    lanes_array.back().get();
    lanes_array.clear();

    /////////////////////////////////////////////////////////
    const int ntest = 100;
    auto begin_timer = iLogger::timestamp_now_float();

    for(int i  = 0; i < ntest; ++i)
        lanes_array = engine->commits(images);
    
    // wait all result
    lanes_array.back().get();

    float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / ntest / images.size();
    auto mode_name = TRT::mode_string(mode);
    INFO("%s[CLRNet] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), inference_average_time, 1000 / inference_average_time);
    append_to_file("perf.result.log", iLogger::format("%s,CLRNet,%s,%f", model_name.c_str(), mode_name, inference_average_time));

    string root = iLogger::format("%s_CLRNet_%s_result", model_name.c_str(), mode_name);
    iLogger::rmtree(root);
    iLogger::mkdir(root);
    int cut_height = 270;
    for(int i = 0; i < lanes_array.size(); ++i){

        auto& image = images[i];
        auto lanes  = lanes_array[i].get();
        for(auto& lane : lanes){
            for(auto& point : lane.points){
                float factor = cut_height / (float)(image.rows);
                float lane_y = (1 - factor) * point.y + factor;
                int x = (int)(point.x * image.cols);
                int y = (int)(lane_y * image.rows);
                cv::circle(image, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
            }
        }

        string file_name = iLogger::file_name(files[i], false);
        string save_path = iLogger::format("%s/%s.jpg", root.c_str(), file_name.c_str());
        INFO("Save to %s, %d lanes, average time %.2f ms", save_path.c_str(), lanes.size(), inference_average_time);
        cv::imwrite(save_path, image);                
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
            CLRNet::image_to_tensor(image, tensor, i, 270);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test CLRNet %s %s ==================================", mode_name, name);      

    if(not requires(name))
        return;
    
    string onnx_file  = iLogger::format("%s.onnx", name);
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

static void test_video(){

    auto engine = CLRNet::create_infer(
        "clrnet.sim.FP16.trtmodel",             // engine file
        0,                                      // gpu id
        0.4f,                                   // confidence threshold
        50.0f,                                  // nms threshold
        4,                                      // nms topk
        270,                                    // cut height
        CLRNet::NMSMethod::CPU,                 // NMS method, fast GPU / CPU
        256,                                    // max lanes
        false                                   // preprocess use multi stream
    );
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    cv::Mat frame;
    cv::VideoCapture cap("inference_lane/videos/solidWhiteRight.mp4");
    if(!cap.isOpened()){
        INFOE("Camera open failed");
        return;
    }    

    // auto remote_show = create_zmq_remote_show();
    while(cap.read(frame)){
        
        auto t0 = iLogger::timestamp_now_float();
        auto lanes = engine->commit(frame).get();
        int cut_height = 270;
        for(auto& lane : lanes){
            for(auto& point : lane.points){
                // INFO("point.x = %.2f, point.y = %.2f", point.x, point.y);
                float factor = cut_height / (float)(frame.rows);
                float lane_y = (1 - factor) * point.y + factor;
                int x = (int)(point.x * frame.cols);
                int y = (int)(lane_y * frame.rows);
                cv::circle(frame, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
            }            
        }
        auto fee = iLogger::timestamp_now_float() - t0;
        INFO("fee = %.2f ms, FPS = %.2f", fee, 1 / fee * 1000);
        // remote_show->post(frame);
        cv::imshow("frame", frame);
        int key = cv::waitKey(1);
        if(key == 27){
            break;
        }
    }

    // char end_signal = 'x';
    // remote_show->post(&end_signal, 1);
    INFO("Done");
    cap.release();
    cv::destroyAllWindows();
    engine.reset();    
}

static void test_single_image(){

    auto engine = CLRNet::create_infer(
        "clrnet.sim.FP16.trtmodel",             // engine file
        0,                                      // gpu id
        0.4f,                                   // confidence threshold
        50.0f,                                  // nms threshold
        4,                                      // nms topk
        270,                                    // cut height
        CLRNet::NMSMethod::FastGPU,             // NMS method, fast GPU / CPU
        128,                                    // max lanes
        false                                   // preprocess use multi stream
    );
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    cv::Mat image = cv::imread("inference_lane/images/02610.jpg");
    if(image.empty()){
        INFOE("Image is empty");
        return;
    }

    auto lanes = engine->commit(image).get();
    int cut_height = 270;
    for(auto& lane : lanes){
        for(auto& point : lane.points){
            float factor = cut_height / (float)(image.rows);
            float lane_y = (1 - factor) * point.y + factor;
            int x = (int)(point.x * image.cols);
            int y = (int)(lane_y * image.rows);
            cv::circle(image, cv::Point(x, y), 3, cv::Scalar(0, 255, 0), -1);
        }
    }
    INFO("Save to clrnet_result.jpg, %d lanes", lanes.size());
    cv::imwrite("clrnet_result.jpg", image);
    engine.reset();
}

int app_clrnet(){

    test(TRT::Mode::FP16, "clrnet.sim");
    // test_single_image();
    // test_video();
    return 0;
}