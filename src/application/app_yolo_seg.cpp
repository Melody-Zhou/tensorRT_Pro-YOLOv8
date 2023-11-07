
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_yolo_seg/yolo_seg.hpp"
#include "tools/zmq_remote_show.hpp"

using namespace std;

static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
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

static void draw_mask(cv::Mat& image, YoloSeg::Box& obj, cv::Scalar& color){

    // compute IM
    float scale_x = 640 / (float)image.cols;
    float scale_y = 640 / (float)image.rows;
    float scale   = std::min(scale_x, scale_y);
    float ox      = -scale * image.cols * 0.5 + 640 * 0.5 + scale * 0.5 - 0.5;
    float oy      = -scale * image.rows * 0.5 + 640 * 0.5 + scale * 0.5 - 0.5;
    cv::Mat M     = (cv::Mat_<float>(2, 3)<<scale, 0, ox, 0, scale, oy);    
    
    cv::Mat IM;
    cv::invertAffineTransform(M, IM); 

    cv::Mat mask_map = cv::Mat::zeros(cv::Size(160, 160), CV_8UC1);
    cv::Mat small_mask(obj.seg->height, obj.seg->width, CV_8UC1, obj.seg->data);
    cv::Rect roi(obj.seg->left, obj.seg->top, obj.seg->width, obj.seg->height);
    small_mask.copyTo(mask_map(roi));
    cv::resize(mask_map, mask_map, cv::Size(640, 640)); // 640x640
    cv::threshold(mask_map, mask_map, 128, 1, cv::THRESH_BINARY);

    cv::Mat mask_resized;
    cv::warpAffine(mask_map, mask_resized, IM, image.size(), cv::INTER_LINEAR);

    // create color mask
    cv::Mat colored_mask = cv::Mat::ones(image.size(), CV_8UC3);
    colored_mask.setTo(color);

    cv::Mat masked_colored_mask;
    cv::bitwise_and(colored_mask, colored_mask, masked_colored_mask, mask_resized);

    // create mask indices
    cv::Mat mask_indices;
    cv::compare(mask_resized, 1, mask_indices, cv::CMP_EQ);
    
    cv::Mat image_masked, colored_mask_masked;
    image.copyTo(image_masked, mask_indices);
    masked_colored_mask.copyTo(colored_mask_masked, mask_indices);

    // weighted sum
    cv::Mat result_masked;
    cv::addWeighted(image_masked, 0.6, colored_mask_masked, 0.4, 0, result_masked);
    
    // copy result to image
    result_masked.copyTo(image, mask_indices);
}

static void inference_and_performance(int deviceid, const string& engine_file, TRT::Mode mode, const string& model_name){

    auto engine = YoloSeg::create_infer(
        engine_file,                // engine file
        deviceid,                   // gpu id
        0.25f,                      // confidence threshold
        0.45f,                      // nms threshold
        YoloSeg::NMSMethod::FastGPU,   // NMS method, fast GPU / CPU
        1024,                       // max objects
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
    vector<shared_future<YoloSeg::BoxArray>> boxes_array;
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
    INFO("%s[YoloV8-Seg] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), inference_average_time, 1000 / inference_average_time);
    append_to_file("perf.result.log", iLogger::format("%s,YoloV8-Seg,%s,%f", model_name.c_str(), mode_name, inference_average_time));

    string root = iLogger::format("%s_YoloV8-Seg_%s_result", model_name.c_str(), mode_name);
    iLogger::rmtree(root);
    iLogger::mkdir(root);

    for(int i = 0; i < boxes_array.size(); ++i){

        auto& image = images[i];
        auto boxes  = boxes_array[i].get();
        
        for(auto& obj : boxes){
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(obj.class_label);        
            cv::Scalar color(b, g, r);
            if(obj.seg){
                draw_mask(image, obj, color);
            }
        }

        for(auto& obj : boxes){
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(obj.class_label);        
            cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

            auto name    = cocolabels[obj.class_label];
            auto caption = iLogger::format("%s %.2f", name, obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
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
            YoloSeg::image_to_tensor(image, tensor, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test YoloV8-Seg %s %s ==================================", mode_name, name);

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
    
    auto engine = YoloSeg::create_infer(
        "yolov8s-seg.FP32.trtmodel",            // engine file
        0,                                      // gpu id
        0.25f,                                  // confidence threshold
        0.45f,                                  // nms threshold
        YoloSeg::NMSMethod::FastGPU,            // NMS method, fast GPU / CPU
        1024,                                   // max objects
        false                                   // preprocess use multi stream
    );
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    cv::Mat frame;
    // cv::VideoCapture cap("exp/face_tracker.mp4");
    cv::VideoCapture cap(0);    // usb camera
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
            tie(b, g, r) = iLogger::random_color(obj.class_label);
            cv::Scalar color(b, g, r);
            if(obj.seg){
                draw_mask(frame, obj, color);
            }
        }
        for(auto& obj : boxes){
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(obj.class_label);
            cv::rectangle(frame, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

            auto name    = cocolabels[obj.class_label];
            auto caption = iLogger::format("%s %.2f", name, obj.confidence);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(frame, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
            cv::putText(frame, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
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
    
    auto engine = YoloSeg::create_infer(
        "yolov8s-seg.FP32.trtmodel",            // engine file
        0,                                      // gpu id
        0.25f,                                  // confidence threshold
        0.45f,                                  // nms threshold
        YoloSeg::NMSMethod::FastGPU,            // NMS method, fast GPU / CPU
        1024,                                   // max objects
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

    auto boxes = engine->commit(image).get();

    int i = 0;
    for(auto& obj : boxes){
        uint8_t b, g, r;
        tie(b, g, r) = iLogger::random_color(obj.class_label);
        cv::Scalar color(b, g, r);
        if(obj.seg){
            draw_mask(image, obj, color);
            cv::imwrite(iLogger::format("%d_mask.jpg", i),
                        cv::Mat(obj.seg->height, obj.seg->width, CV_8U, obj.seg->data));
            i++;
        }
    }

    for(auto& obj : boxes){
        uint8_t b, g, r;
        tie(b, g, r) = iLogger::random_color(obj.class_label);
        cv::rectangle(image, cv::Point(obj.left, obj.top), cv::Point(obj.right, obj.bottom), cv::Scalar(b, g, r), 5);

        auto name    = cocolabels[obj.class_label];
        auto caption = iLogger::format("%s %.2f", name, obj.confidence);
        int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
        cv::rectangle(image, cv::Point(obj.left-3, obj.top-33), cv::Point(obj.left + width, obj.top), cv::Scalar(b, g, r), -1);
        cv::putText(image, caption, cv::Point(obj.left, obj.top-5), 0, 1, cv::Scalar::all(0), 2, 16);
    }
    INFO("Save to Result-seg.jpg, %d objects", boxes.size());
    cv::imwrite("Result-seg.jpg", image);
    engine.reset();
}

int app_yolo_seg(){
 
    test(TRT::Mode::FP32, "yolov8s-seg");
    // test_single_image();
    // test_video();
    return 0;
}