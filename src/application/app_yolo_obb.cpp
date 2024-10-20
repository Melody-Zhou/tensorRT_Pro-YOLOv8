
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_yolo_obb/yolo_obb.hpp"
#include "tools/zmq_remote_show.hpp"

using namespace std;

static const char* dotalabels[] = {
    "plane", "ship", "storage tank", "baseball diamond", "tennis court",
    "basketball court", "ground track field", "harbor", "bridge", "large vehicle",
    "small vehicle", "helicopter", "roundabout", "soccer ball field", "swimming pool"
};

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

static vector<cv::Point> xywhr2xyxyxyxy(const YoloOBB::Box& box) {
    float cos_value = std::cos(box.angle);
    float sin_value = std::sin(box.angle);

    float w_2 = box.width / 2, h_2 = box.height / 2;
    float vec1_x =  w_2 * cos_value, vec1_y = w_2 * sin_value;
    float vec2_x = -h_2 * sin_value, vec2_y = h_2 * cos_value;

    vector<cv::Point> corners;
    corners.push_back(cv::Point(box.center_x + vec1_x + vec2_x, box.center_y + vec1_y + vec2_y));
    corners.push_back(cv::Point(box.center_x + vec1_x - vec2_x, box.center_y + vec1_y - vec2_y));
    corners.push_back(cv::Point(box.center_x - vec1_x - vec2_x, box.center_y - vec1_y - vec2_y));
    corners.push_back(cv::Point(box.center_x - vec1_x + vec2_x, box.center_y - vec1_y + vec2_y));

    return corners;
}

static void inference_and_performance(int deviceid, const string& engine_file, TRT::Mode mode, const string& model_name){
    
    auto engine = YoloOBB::create_infer(
        engine_file,                   // engine file
        deviceid,                      // gpu id
        0.25f,                         // confidence threshold
        0.45f,                         // nms threshold
        YoloOBB::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
        1024,                          // max objects
        false                          // preprocess use multi stream
    );
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    auto files = iLogger::find_files("inference_obb", "*.jpg;*.jpeg;*.png;*.gif;*.tif");    
    vector<cv::Mat> images;
    for(int i = 0; i < files.size(); ++i){
        auto image = cv::imread(files[i]);
        images.emplace_back(image);
    }

    // warmup
    vector<shared_future<YoloOBB::BoxArray>> boxes_array;
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
    INFO("%s[YoloV8-OBB] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), inference_average_time, 1000 / inference_average_time);
    append_to_file("perf.result.log", iLogger::format("%s,YoloV8-OBB,%s,%f", model_name.c_str(), mode_name, inference_average_time));

    string root = iLogger::format("%s_YoloV8-OBB_%s_result", model_name.c_str(), mode_name);
    iLogger::rmtree(root);
    iLogger::mkdir(root);

    for(int i = 0; i < boxes_array.size(); ++i){

        auto& image = images[i];
        auto boxes  = boxes_array[i].get();
        
        for(auto& obj : boxes){
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(obj.class_label);
            auto corners = xywhr2xyxyxyxy(obj);
            cv::polylines(image, vector<vector<cv::Point>>{corners}, true, cv::Scalar(b, g, r), 2, 16);

            auto name    = dotalabels[obj.class_label];
            auto caption = iLogger::format("%s %.2f", name, obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(corners[0].x-3, corners[0].y-33), cv::Point(corners[0].x-3 + width, corners[0].y), cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(corners[0].x-3, corners[0].y-5), 0, 1, cv::Scalar::all(0), 2, 16);
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
            YoloOBB::image_to_tensor(image, tensor, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test YoloV8-OBB %s %s ==================================", mode_name, name);

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
    
    auto engine = YoloOBB::create_infer(
        "yolov8s-obb.FP32.trtmodel",            // engine file
        0,                                      // gpu id
        0.25f,                                  // confidence threshold
        0.45f,                                  // nms threshold
        YoloOBB::NMSMethod::FastGPU,            // NMS method, fast GPU / CPU
        1024,                                   // max objects
        false                                   // preprocess use multi stream
    );    
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    cv::Mat image = cv::imread("inference_obb/P0032.jpg");
    if(image.empty()){
        INFOE("Image is empty");
        return;
    }    

    auto boxes = engine->commit(image).get();

    for(auto& obj : boxes){
        uint8_t b, g, r;
        tie(b, g, r) = iLogger::random_color(obj.class_label);
        auto corners = xywhr2xyxyxyxy(obj);
        cv::polylines(image, vector<vector<cv::Point>>{corners}, true, cv::Scalar(b, g, r), 2, 16);

        auto name = dotalabels[obj.class_label];
        auto caption = iLogger::format("%s %.2f", name, obj.confidence);
        int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(corners[0].x-3, corners[0].y-33), cv::Point(corners[0].x-3 + width, corners[0].y), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(corners[0].x-3, corners[0].y-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    INFO("Save to Result.jpg, %d objects", boxes.size());
    cv::imwrite("Result.jpg", image);
    engine.reset();
}

void perf() {
    int max_infer_batch = 16;
    int batch = 16;
    std::vector<cv::Mat> images{cv::imread("inference_obb/P0009.jpg")};

    for (int i = images.size(); i < batch; ++i) images.push_back(images[i % 1]);

    auto engine = YoloOBB::create_infer("yolov8s-obb.FP32.trtmodel", 0);

    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    // warmup
    YoloOBB::BoxArray boxes;
    vector<shared_future<YoloOBB::BoxArray>> boxes_array;
    for(int i = 0; i < 10; ++i)
        boxes_array = engine->commits(images);
    boxes_array.back().get();
    boxes_array.clear();

    const int ntest = 100;
    INFO("warmup done.");
    INFO("begine test BATCH1 and BATCH16, ntest = %d", ntest);
    
    // BATCH1
    auto begin_timer_b1 = iLogger::timestamp_now_float();
    for(int i = 0; i < ntest; ++i) {
        boxes = engine->commit(images[0]).get();
    }
    float inference_average_time_b1 = (iLogger::timestamp_now_float() - begin_timer_b1) / ntest;
    
    // BATCH16
    auto begin_timer_b16 = iLogger::timestamp_now_float();
    for (int i = 0; i < ntest; ++i) {
        boxes_array = engine->commits(images);
    }
    boxes_array.back().get();
    float inference_average_time_b16 = (iLogger::timestamp_now_float() - begin_timer_b16) / ntest / images.size();

    INFO("BATCH1:  %.2f ms", inference_average_time_b1);
    INFO("BATCH16: %.2f ms", inference_average_time_b16);
}

int app_yolo_obb(){
 
    test(TRT::Mode::FP32, "yolov8s-obb");
    // test(TRT::Mode::FP32, "yolo11s-obb");
    // test_single_image();
    // perf();

    return 0;
}