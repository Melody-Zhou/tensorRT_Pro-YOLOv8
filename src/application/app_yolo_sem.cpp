#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_yolo_sem/yolo_sem.hpp"

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

static cv::Mat semantic_overlay(const cv::Mat& image, const cv::Mat& class_map, float alpha=0.5f){

    // Ultralytics Colors palette, hex2rgb then BGR:
    // hexs = {"042AFF","0BDBEB","F3F3F3","00DFB7","111F68","FF6FDD","FF444F","CCED00","00F344",
    //         "BD00FF","00B4FF","DD00BA","00FFFF","26C000","01FFB3","7D24FF","7B0068","FF1B6C","FC6D2F","A2FF0B"}
    static const cv::Scalar class_colors[] = {
        cv::Scalar(255, 42, 4),     // 0  #042AFF -> BGR
        cv::Scalar(235, 219, 11),   // 1  #0BDBEB -> BGR
        cv::Scalar(243, 243, 243),  // 2  #F3F3F3 -> BGR
        cv::Scalar(183, 223, 0),    // 3  #00DFB7 -> BGR
        cv::Scalar(104, 31, 17),    // 4  #111F68 -> BGR
        cv::Scalar(221, 111, 255),  // 5  #FF6FDD -> BGR
        cv::Scalar(79, 68, 255),    // 6  #FF444F -> BGR
        cv::Scalar(0, 237, 204),    // 7  #CCED00 -> BGR
        cv::Scalar(68, 243, 0),     // 8  #00F344 -> BGR
        cv::Scalar(255, 0, 189),    // 9  #BD00FF -> BGR
        cv::Scalar(255, 180, 0),    // 10 #00B4FF -> BGR
        cv::Scalar(186, 0, 221),    // 11 #DD00BA -> BGR
        cv::Scalar(255, 255, 0),    // 12 #00FFFF -> BGR
        cv::Scalar(0, 192, 38),     // 13 #26C000 -> BGR
        cv::Scalar(179, 255, 1),    // 14 #01FFB3 -> BGR
        cv::Scalar(255, 36, 125),   // 15 #7D24FF -> BGR
        cv::Scalar(104, 0, 123),    // 16 #7B0068 -> BGR
        cv::Scalar(108, 27, 255),   // 17 #FF1B6C -> BGR
        cv::Scalar(47, 109, 252),   // 18 #FC6D2F -> BGR
        cv::Scalar(11, 255, 162),   // 19 #A2FF0B -> BGR
    };
    const int num_colors = sizeof(class_colors) / sizeof(class_colors[0]);

    cv::Mat overlay = cv::Mat::zeros(image.size(), image.type());
    for(int y = 0; y < class_map.rows; ++y){
        for(int x = 0; x < class_map.cols; ++x){
            uint8_t cls_id = class_map.at<uint8_t>(y, x);
            if(cls_id == 255) continue;
            int idx = cls_id % num_colors;
            overlay.at<cv::Vec3b>(y, x) = cv::Vec3b(
                (uchar)(class_colors[idx][0]),
                (uchar)(class_colors[idx][1]),
                (uchar)(class_colors[idx][2])
            );
        }
    }

    cv::Mat result;
    cv::addWeighted(image, 1 - alpha, overlay, alpha, 0, result);
    return result;
}

static void save_semantic_result(const cv::Mat& image, const cv::Mat& class_map, const string& save_path, float alpha=0.5f){
    cv::Mat result = semantic_overlay(image, class_map, alpha);
    cv::imwrite(save_path, result);
    INFO("Saved to %s", save_path.c_str());
}

static void inference_and_performance(int deviceid, const string& engine_file, TRT::Mode mode, const string& model_name){

    auto engine = YoloSEM::create_infer(
        engine_file,                                // engine file
        deviceid,                                   // gpu id
        false                                       // preprocess use multi stream
    );

    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    auto files = iLogger::find_files("inference_sem", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    vector<cv::Mat> images;
    for(int i = 0; i < files.size(); ++i){
        auto image = cv::imread(files[i]);
        images.emplace_back(image);
    }

    // warmup
    vector<shared_future<cv::Mat>> class_map_array;
    for(int i = 0; i < 10; ++i)
        class_map_array = engine->commits(images);
    class_map_array.back().get();
    class_map_array.clear();

    /////////////////////////////////////////////////////////
    const int ntest = 100;
    auto begin_timer = iLogger::timestamp_now_float();

    for(int i  = 0; i < ntest; ++i)
        class_map_array = engine->commits(images);

    // wait all result
    class_map_array.back().get();

    float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / ntest / images.size();
    auto mode_name = TRT::mode_string(mode);
    INFO("%s[%s] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), mode_name, inference_average_time, 1000 / inference_average_time);
    append_to_file("perf.result.log", iLogger::format("%s,%s,%f", model_name.c_str(), mode_name, inference_average_time));

    string root = iLogger::format("%s_%s_result", model_name.c_str(), mode_name);
    iLogger::rmtree(root);
    iLogger::mkdir(root);

    for(int i = 0; i < class_map_array.size(); ++i){
        auto class_map = class_map_array[i].get();
        string file_name = iLogger::file_name(files[i], false);
        string ext = files[i].substr(files[i].find_last_of(".") + 1);
        string save_path = iLogger::format("%s/%s.%s", root.c_str(), file_name.c_str(), ext.c_str());
        INFO("Save to %s, average time %.2f ms", save_path.c_str(), inference_average_time);
        save_semantic_result(images[i], class_map, save_path);
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
            YoloSEM::image_to_tensor(image, tensor, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test YoloSemantic %s %s ==================================", mode_name, name);

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

static void test_single_image(){

    auto engine = YoloSEM::create_infer(
        "yolo26s-sem.FP16.trtmodel",
        0,
        false
    );

    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    cv::Mat image = cv::imread("inference_sem/munster_000000_000019_leftImg8bit.png");
    if(image.empty()){
        INFOE("Image is empty");
        return;
    }

    auto class_map = engine->commit(image).get();

    save_semantic_result(image, class_map, "semantic_result.jpg");
    engine.reset();
}

static void test_video(){

    auto engine = YoloSEM::create_infer(
        "yolo26s-sem.FP16.trtmodel",
        0,
        false
    );

    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    cv::Mat frame;
    cv::VideoCapture cap("workspace/exp/test.mp4");
    // cv::VideoCapture cap(0);    // usb camera
    if(!cap.isOpened()){
        INFOE("Camera open failed");
        return;
    }

    while(cap.read(frame)){

        auto t0 = iLogger::timestamp_now_float();
        auto class_map = engine->commit(frame).get();
        auto fee = iLogger::timestamp_now_float() - t0;
        INFO("fee = %.2f ms, FPS = %.2f", fee, 1 / fee * 1000);

        // build semantic overlay, hconcat with original frame
        auto semantic = semantic_overlay(frame, class_map);
        cv::Mat result;
        cv::hconcat(frame, semantic, result);
        cv::imshow("YoloSem: Original | Semantic", result);
        int key = cv::waitKey(1);
        if(key == 27)
            break;
    }

    INFO("Done");
    cap.release();
    cv::destroyAllWindows();
    engine.reset();
}

int app_yolo_sem(){

    test(TRT::Mode::FP32, "yolo26s-sem");
    // test_single_image();
    // test_video();

    return 0;
}
