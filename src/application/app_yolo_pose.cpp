
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_yolo_pose/yolo_pose.hpp"
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

static void draw_pose(cv::Mat& image, const vector<cv::Point3f>& keypoints){

    vector<cv::Scalar> pose_palette = {
        {255, 128,   0}, {255, 153,  51}, {255, 178, 102}, {230, 230,   0}, {255, 153, 255},
        {153, 204, 255}, {255, 102, 255}, {255, 51,  255}, {102, 178, 255}, {51,  153, 255},
        {255, 153, 153}, {255, 102, 102}, {255, 51,   51}, {153, 255, 153}, {102, 255, 102},
        {51,  255,  51}, {0,   255,   0}, {0,   0,   255}, {255, 0,     0}, {255, 255, 255}
    };

    vector<cv::Point> skeleton = {
        {15, 13}, {13, 11}, {16, 14}, {14, 12}, {11, 12}, {5, 11}, {6, 12},
        {5,   6}, {5,   7}, {6,   8}, {7,   9}, {8,  10}, {1,  2}, {0,  1}, 
        {0,   2}, {1,   3}, {2,   4}, {3,   5}, {4,   6}
    };
    // 16 0 9
    // 
    vector<cv::Scalar> limb_color = {
        pose_palette[9],  pose_palette[9],  pose_palette[9],  pose_palette[9],  pose_palette[7],
        pose_palette[7],  pose_palette[7],  pose_palette[0],  pose_palette[0],  pose_palette[0],
        pose_palette[0],  pose_palette[0],  pose_palette[16], pose_palette[16], pose_palette[16],
        pose_palette[16], pose_palette[16], pose_palette[16], pose_palette[16]
    };

    vector<cv::Scalar> kpt_color = {
        pose_palette[16], pose_palette[16], pose_palette[16], pose_palette[16], pose_palette[16],
        pose_palette[0],  pose_palette[0],  pose_palette[0],  pose_palette[0],  pose_palette[0],
        pose_palette[0],  pose_palette[9],  pose_palette[9],  pose_palette[9],  pose_palette[9],
        pose_palette[9],  pose_palette[9]
    };

    for(int i = 0; i < keypoints.size(); ++i){
        
        auto& keypoint = keypoints[i];
        if(keypoint.z < 0.5)
            continue;
        if(keypoint.x != 0 && keypoint.y != 0)
            cv::circle(image, cv::Point(keypoint.x, keypoint.y), 5, kpt_color[i], -1, cv::LINE_AA);
    }

    for(int i = 0; i < skeleton.size(); ++i){

        auto& index = skeleton[i];
        auto& pos1  = keypoints[index.x];
        auto& pos2  = keypoints[index.y];

        if(pos1.z < 0.5 || pos2.z < 0.5)
            continue;
        
        if(pos1.x == 0 || pos1.y == 0 || pos2.x == 0 || pos2.y == 0)
            continue;
                
        cv::line(image, cv::Point(pos1.x, pos1.y), cv::Point(pos2.x, pos2.y), limb_color[i], 2, cv::LINE_AA);
    }
}

static void inference_and_performance(int deviceid, const string& engine_file, TRT::Mode mode, const string& model_name){

    auto engine = YoloPose::create_infer(
        engine_file,                    // engine file
        deviceid,                       // gpu id
        0.25f,                          // confidence threshold
        0.45f,                          // nms threshold
        YoloPose::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
        1024,                           // max objects
        false                           // preprocess use multi stream
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
    vector<shared_future<YoloPose::BoxArray>> boxes_array;
    for(int i = 0; i < 10; ++i)
        boxes_array = engine->commits(images);
    boxes_array.back().get();
    boxes_array.clear();
    
    /////////////////////////////////////////////////////////
    const int ntest = 100;
    auto begin_timer = iLogger::timestamp_now_float();

    for(int i  = 0; i < ntest; ++i)
        boxes_array = engine->commits(images);
    
    // wait all result
    boxes_array.back().get();

    float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / ntest / images.size();
    auto mode_name = TRT::mode_string(mode);
    INFO("%s[YoloV8-Pose] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), inference_average_time, 1000 / inference_average_time);
    append_to_file("perf.result.log", iLogger::format("%s,YoloV8-Pose,%s,%f", model_name.c_str(), mode_name, inference_average_time));

    string root = iLogger::format("%s_YoloV8-Pose_%s_result", model_name.c_str(), mode_name);
    iLogger::rmtree(root);
    iLogger::mkdir(root);

    for(int i = 0; i < boxes_array.size(); ++i){

        auto& image = images[i];
        auto boxes  = boxes_array[i].get();
        
        for(auto& obj : boxes){
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(0);
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

            auto caption = iLogger::format("person %.2f", obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
            draw_pose(image, obj.keypoints);
        }

        string file_name = iLogger::file_name(files[i], false);
        string save_path = iLogger::format("%s/%s.jpg", root.c_str(), file_name.c_str());
        INFO("Save to %s, %d object, average time %.2f ms", save_path.c_str(), boxes.size(), inference_average_time);
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
            YoloPose::image_to_tensor(image, tensor, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test YoloV8-Pose %s %s ==================================", mode_name, name);

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

static void test_video(){
    
    auto engine = YoloPose::create_infer(
        "yolov8s-pose.FP32.trtmodel",   // engine file
        0,                              // gpu id
        0.25f,                          // confidence threshold
        0.45f,                          // nms threshold
        YoloPose::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
        1024,                           // max objects
        false                           // preprocess use multi stream
    );
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    cv::Mat frame;
    cv::VideoCapture cap("exp/face_tracker.mp4");
    // cv::VideoCapture cap(0);    // usb camera
    if(!cap.isOpened()){
        INFOE("Camera open failed");
        return;
    }

    // auto remote_show = create_zmq_remote_show();
    while(cap.read(frame)){
        
        auto t0 = iLogger::timestamp_now_float();
        auto boxes = engine->commit(frame).get();
        for(auto& obj : boxes){
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(0);
            cv::rectangle(frame, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

            auto caption = iLogger::format("person %.2f", obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(frame, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(frame, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
            draw_pose(frame, obj.keypoints);
        }
        auto fee = iLogger::timestamp_now_float() - t0;
        INFO("fee = %.2f ms, FPS = %.2f", fee, 1 / fee * 1000);
        // remote_show->post(frame);
        cv::imshow("frame", frame);
        int key = cv::waitKey(1);
        if(key == 27)
            break;
    }
    
    // char end_signal = 'x';
    // remote_show->post(&end_signal, 1);
    INFO("Done");
    cap.release();
    cv::destroyAllWindows();
    engine.reset();
}

static void test_single_image(){
    
    auto engine = YoloPose::create_infer(
        "yolov8s-pose.FP32.trtmodel",   // engine file
        0,                              // gpu id
        0.25f,                          // confidence threshold
        0.45f,                          // nms threshold
        YoloPose::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
        1024,                           // max objects
        false                           // preprocess use multi stream
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

    auto boxes = engine->commit(image).get();

    for(auto& obj : boxes){
        uint8_t b, g, r;
        tie(b, g, r) = iLogger::random_color(0);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

        auto caption = iLogger::format("person %.2f", obj.confidence);
        int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
        draw_pose(image, obj.keypoints);
    }
    INFO("Save to Result-pose.jpg, %d objects", boxes.size());
    cv::imwrite("Result-pose.jpg", image);
    engine.reset();
}

int app_yolo_pose(){
 
    test(TRT::Mode::FP32, "yolov8s-pose");
    // test_single_image();
    // test_video();
    return 0;
}