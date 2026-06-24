#include <unordered_map>
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_rfdetr/rfdetr.hpp"
#include "tools/zmq_remote_show.hpp"

using namespace std;

// COCO class ID -> class name map (same as rfdetr_infer.py COCO_CLASSES)
static const unordered_map<int, string> coco_classes = {
    {1,  "person"}, {2,  "bicycle"}, {3,  "car"}, {4,  "motorcycle"}, {5,  "airplane"},
    {6,  "bus"}, {7,  "train"}, {8,  "truck"}, {9,  "boat"}, {10, "traffic light"},
    {11, "fire hydrant"}, {13, "stop sign"}, {14, "parking meter"}, {15, "bench"},
    {16, "bird"}, {17, "cat"}, {18, "dog"}, {19, "horse"}, {20, "sheep"},
    {21, "cow"}, {22, "elephant"}, {23, "bear"}, {24, "zebra"}, {25, "giraffe"},
    {27, "backpack"}, {28, "umbrella"}, {31, "handbag"}, {32, "tie"},
    {33, "suitcase"}, {34, "frisbee"}, {35, "skis"}, {36, "snowboard"},
    {37, "sports ball"}, {38, "kite"}, {39, "baseball bat"}, {40, "baseball glove"},
    {41, "skateboard"}, {42, "surfboard"}, {43, "tennis racket"}, {44, "bottle"},
    {46, "wine glass"}, {47, "cup"}, {48, "fork"}, {49, "knife"}, {50, "spoon"},
    {51, "bowl"}, {52, "banana"}, {53, "apple"}, {54, "sandwich"}, {55, "orange"},
    {56, "broccoli"}, {57, "carrot"}, {58, "hot dog"}, {59, "pizza"}, {60, "donut"},
    {61, "cake"}, {62, "chair"}, {63, "couch"}, {64, "potted plant"}, {65, "bed"},
    {67, "dining table"}, {70, "toilet"}, {72, "tv"}, {73, "laptop"},
    {74, "mouse"}, {75, "remote"}, {76, "keyboard"}, {77, "cell phone"},
    {78, "microwave"}, {79, "oven"}, {80, "toaster"}, {81, "sink"},
    {82, "refrigerator"}, {84, "book"}, {85, "clock"}, {86, "vase"},
    {87, "scissors"}, {88, "teddy bear"}, {89, "hair drier"}, {90, "toothbrush"}
};

static const char* get_coco_class_name(int cls_id){
    auto it = coco_classes.find(cls_id);
    if(it != coco_classes.end())
        return it->second.c_str();
    static char buf[16];
    snprintf(buf, sizeof(buf), "%d", cls_id);
    return buf;
}

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

    auto engine = RFDETR::create_infer(
        engine_file,                // engine file
        deviceid,                   // gpu id
        0.5f,                       // confidence threshold
        300,                        // max objects
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
    vector<shared_future<RFDETR::BoxArray>> boxes_array;
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
    INFO("%s[RF-DETR] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), inference_average_time, 1000 / inference_average_time);
    append_to_file("perf.result.log", iLogger::format("%s,RF-DETR,%s,%f", model_name.c_str(), mode_name, inference_average_time));

    string root = iLogger::format("%s_RF-DETR_%s_result", model_name.c_str(), mode_name);
    iLogger::rmtree(root);
    iLogger::mkdir(root);

    for(int i = 0; i < boxes_array.size(); ++i){

        auto& image = images[i];
        int ow = image.cols;
        int oh = image.rows;
        auto boxes  = boxes_array[i].get();

        for(auto& obj : boxes){
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(obj.class_label);
            float left   = obj.left * ow;
            float top    = obj.top  * oh;
            float right  = obj.right  * ow;
            float bottom = obj.bottom * oh;
            cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(b, g, r), 5);

            auto name    = get_coco_class_name(obj.class_label);
            auto caption = iLogger::format("%s %.2f", name, obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(left-3, top-33), cv::Point(left + width, top), cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
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
            RFDETR::image_to_tensor(image, tensor, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test RF-DETR %s %s ==================================", mode_name, name);

    if(not requires(name))
        return;

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 1;

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

    auto engine = RFDETR::create_infer(
        "rfdetr-medium.FP32.trtmodel",          // engine file
        0,                                      // gpu id
        0.5f,                                   // confidence threshold
        300,                                    // max objects
        false                                   // preprocess use multi stream
    );
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    cv::Mat frame;
    cv::VideoCapture cap("exp/test.mp4");
    // cv::VideoCapture cap(0);    // usb camera
    if(!cap.isOpened()){
        INFOE("Camera open failed");
        return;
    }

    // auto remote_show = create_zmq_remote_show();
    while(cap.read(frame)){

        auto t0 = iLogger::timestamp_now_float();
        auto boxes = engine->commit(frame).get();
        int ow = frame.cols;
        int oh = frame.rows;
        for(auto& obj : boxes){
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(obj.class_label);

            float left   = obj.left * ow;
            float top    = obj.top  * oh;
            float right  = obj.right  * ow;
            float bottom = obj.bottom * oh;
            cv::rectangle(frame, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(b, g, r), 5);

            auto name    = get_coco_class_name(obj.class_label);
            auto caption = iLogger::format("%s %.2f", name, obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(frame, cv::Point(left-3, top-33), cv::Point(left + width, top), cv::Scalar(b, g, r), -1);
            cv::putText(frame, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
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

    auto engine = RFDETR::create_infer(
        "rfdetr-medium.FP32.trtmodel",          // engine file
        0,                                      // gpu id
        0.5f,                                   // confidence threshold
        300,                                    // max objects
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

    int ow = image.cols;
    int oh = image.rows;

    auto boxes = engine->commit(image).get();

    for(auto& obj : boxes){
        uint8_t b, g, r;
        tie(b, g, r) = iLogger::random_color(obj.class_label);
        float left   = obj.left * ow;
        float top    = obj.top  * oh;
        float right  = obj.right  * ow;
        float bottom = obj.bottom * oh;
        cv::rectangle(image, cv::Point(left, top), cv::Point(right, bottom), cv::Scalar(b, g, r), 5);

        auto name    = get_coco_class_name(obj.class_label);
        auto caption = iLogger::format("%s %.2f", name, obj.confidence);
        int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(left-3, top-33), cv::Point(left + width, top), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(left, top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    INFO("Save to Result-RF-DETR.jpg, %d objects", boxes.size());
    cv::imwrite("Result-RF-DETR.jpg", image);
    engine.reset();
}

int app_rfdetr(){

    /*
    注意：
    1. RF-DETR 在 TensorRT 8.6 的 FP16 精度下推理存在异常，这并非后处理塞进计算图的问题，
       而是 RF-DETR 的 DINOv2 backbone + Transformer 在 TensorRT 8.6 的 FP16 计算路径本身产生了错误输出。
    2. 如果你想使用 FP16 精度，请使用 TensorRT 10.x 或以上版本。
    3. 更多细节信息请参考：https://github.com/roboflow/rf-detr/issues/352   
    */

    test(TRT::Mode::FP32, "rfdetr-medium");
    // test_single_image();
    // test_video();
    return 0;
}
