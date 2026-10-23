#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include "app_yolo_depth/yolo_depth.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

using namespace std;

static void append_to_file(const string& file, const string& data){
    FILE* f = fopen(file.c_str(), "a+");
    if(f == nullptr){
        INFOE("Open %s failed.", file.c_str());
        return;
    }

    fprintf(f, "%s\n", data.c_str());
    fclose(f);
}

enum class DepthColorMap{
    Jet,
    Inferno,
    Spectral
};

// Spectral_r anchors (RGB, far -> near), same as ultralytics colorize_depth
static const cv::Mat& spectral_lut(){

    static const cv::Mat lut = [](){
        static const float anchors[11][3] = {
            { 94,  79, 162}, { 51, 135, 188}, {102, 194, 165}, {170, 220, 164},
            {230, 245, 152}, {255, 254, 190}, {254, 224, 139}, {253, 173,  96},
            {244, 109,  67}, {212,  61,  79}, {158,   1,  66}
        };

        cv::Mat table(256, 1, CV_8UC3);
        for(int k = 0; k < 256; ++k){
            float x = k * 10.0f / 255.0f;
            int   i = std::min((int)x, 9);
            float f = x - i;
            for(int c = 0; c < 3; ++c){
                float rgb = anchors[i][c] * (1.0f - f) + anchors[i + 1][c] * f;
                // RGB -> BGR for OpenCV
                table.at<cv::Vec3b>(k, 0)[2 - c] = (uchar)std::nearbyint(rgb);
            }
        }
        return table;
    }();
    return lut;
}

static float percentile_sorted(const vector<float>& values, float p){
    if(values.empty())
        return 0.0f;

    if(values.size() == 1)
        return values[0];

    float position = p * (values.size() - 1);
    size_t low     = (size_t)std::floor(position);
    size_t high    = (size_t)std::ceil(position);
    float t        = position - low;
    return values[low] * (1.0f - t) + values[high] * t;
}

static void disparity_range(const cv::Mat& depth, float& vmin, float& vmax){

    vector<float> values;
    values.reserve(depth.total());
    for(int y = 0; y < depth.rows; ++y){
        const float* row = depth.ptr<float>(y);
        for(int x = 0; x < depth.cols; ++x){
            float d = row[x];
            if(d > 0)
                values.push_back(1.0f / d);
        }
    }

    if(values.empty()){
        vmin = 0.0f;
        vmax = 1.0f;
        return;
    }

    std::sort(values.begin(), values.end());
    vmin = percentile_sorted(values, 0.02f);
    vmax = percentile_sorted(values, 0.98f);
    if(vmax <= vmin)
        vmax = vmin + 1e-6f;
}

static cv::Mat colorize_depth(
    const cv::Mat& depth, bool disparity,
    float vmin, float vmax, DepthColorMap cmap
){    if(vmax <= vmin)
        vmax = vmin + 1e-6f;

    cv::Mat index(depth.size(), CV_8UC1);
    cv::Mat valid_mask(depth.size(), CV_8UC1);

    for(int y = 0; y < depth.rows; ++y){
        const float* src = depth.ptr<float>(y);
        uchar* idx       = index.ptr<uchar>(y);
        uchar* mask      = valid_mask.ptr<uchar>(y);
        for(int x = 0; x < depth.cols; ++x){
            float d     = src[x];
            bool  valid = d > 0;
            mask[x]     = valid ? 255 : 0;
            if(!valid){
                idx[x] = 0;
                continue;
            }

            float v = disparity ? 1.0f / d : d;
            float t = (v - vmin) / (vmax - vmin);
            t = t < 0.0f ? 0.0f : (t > 1.0f ? 1.0f : t);
            idx[x] = (uchar)(t * 255.0f);
        }
    }

    cv::Mat color;
    if(cmap == DepthColorMap::Spectral)
        cv::applyColorMap(index, color, spectral_lut());
    else if(cmap == DepthColorMap::Jet)
        cv::applyColorMap(index, color, cv::COLORMAP_JET);
    else
        cv::applyColorMap(index, color, cv::COLORMAP_INFERNO);

    cv::Mat invalid_mask = (valid_mask == 0);
    color.setTo(cv::Scalar(0, 0, 0), invalid_mask);
    return color;
}

static cv::Mat depth_overlay(
    const cv::Mat& image, const cv::Mat& depth,
    float vmin, float vmax, float alpha = 0.6f
){
    cv::Mat heat = colorize_depth(depth, true, vmin, vmax, DepthColorMap::Jet);
    cv::Mat result;
    cv::addWeighted(image, 1.0f - alpha, heat, alpha, 0, result);
    return result;
}

static void inference_and_performance(int deviceid, const string& engine_file, TRT::Mode mode, const string& model_name){

    auto engine = YoloDepth::create_infer(
        engine_file,                                // engine file
        deviceid,                                   // gpu id
        false                                       // preprocess use multi stream
    );

    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    auto files = iLogger::find_files("inference_depth/images", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    vector<string> valid_files;
    vector<cv::Mat> images;
    for(int i = 0; i < files.size(); ++i){
        auto image = cv::imread(files[i]);
        if(image.empty()){
            INFOE("Load image failed: %s", files[i].c_str());
            continue;
        }
        valid_files.emplace_back(files[i]);
        images.emplace_back(image);
    }

    if(images.empty()){
        INFOE("No images found in inference_depth/images");
        engine.reset();
        return;
    }

    // warmup
    vector<shared_future<cv::Mat>> depth_array;
    for(int i = 0; i < 10; ++i)
        depth_array = engine->commits(images);
    depth_array.back().get();
    depth_array.clear();

    /////////////////////////////////////////////////////////
    const int ntest = 100;
    auto begin_timer = iLogger::timestamp_now_float();

    for(int i = 0; i < ntest; ++i)
        depth_array = engine->commits(images);

    // wait all result
    depth_array.back().get();

    float inference_average_time = (iLogger::timestamp_now_float() - begin_timer) / ntest / images.size();
    auto mode_name = TRT::mode_string(mode);
    INFO("%s[%s] average: %.2f ms / image, FPS: %.2f", engine_file.c_str(), mode_name, inference_average_time, 1000.0f / inference_average_time);
    append_to_file("perf.result.log", iLogger::format("%s,%s,%f", model_name.c_str(), mode_name, inference_average_time));

    string root = iLogger::format("%s_%s_result", model_name.c_str(), mode_name);
    iLogger::rmtree(root);
    iLogger::mkdir(root);

    for(int i = 0; i < depth_array.size(); ++i){
        auto depth = depth_array[i].get();
        if(depth.empty()){
            INFOE("Depth result %d is empty", i);
            continue;
        }

        string file_name = iLogger::file_name(valid_files[i], false);

        float vmin = 0.0f, vmax = 1.0f;
        disparity_range(depth, vmin, vmax);

        // near = warm, robust to far outliers, same as infer.py colorize_depth(cmap="spectral")
        string spectral_path = iLogger::format("%s/%s_depth.jpg", root.c_str(), file_name.c_str());
        cv::imwrite(spectral_path, colorize_depth(depth, true, vmin, vmax, DepthColorMap::Spectral));

        // fixed 0-20 m range so the same color means the same distance across frames
        string metric_path = iLogger::format("%s/%s_metric.jpg", root.c_str(), file_name.c_str());
        cv::imwrite(metric_path, colorize_depth(depth, false, 0.0f, 20.0f, DepthColorMap::Inferno));

        // jet heat overlay on the original image
        cv::Mat overlay = depth_overlay(images[i], depth, vmin, vmax, 0.6f);
        string overlay_path = iLogger::format("%s/%s_overlay.jpg", root.c_str(), file_name.c_str());
        cv::imwrite(overlay_path, overlay);

        // raw metric depth in meters for numeric verification
        string depth_path = iLogger::format("%s/%s_depth.tiff", root.c_str(), file_name.c_str());
        if(!cv::imwrite(depth_path, depth)){
            string bin_path = iLogger::format("%s/%s_depth.bin", root.c_str(), file_name.c_str());
            iLogger::save_file(bin_path, depth.data, depth.total() * sizeof(float));
            INFO("Save to %s (tiff failed, fallback to bin)", bin_path.c_str());
        }else{
            INFO("Save to %s", depth_path.c_str());
        }
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
            if(image.empty())
                continue;
            YoloDepth::image_to_tensor(image, tensor, i);
        }
    };

    const char* name = model.c_str();
    INFO("===================== test YoloDepth %s %s ==================================", mode_name, name);

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 4;

    if(not iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source
            model_file,                 // save to
            int8process,
            "inference"
        );
    }

    inference_and_performance(deviceid, model_file, mode, name);
}

static void test_single_image(){

    auto engine = YoloDepth::create_infer(
        "yolo26s-depth.FP16.trtmodel",              // engine file
        0,                                          // gpu id
        false                                       // preprocess use multi stream
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

    auto depth = engine->commit(image).get();
    if(depth.empty()){
        INFOE("Depth is empty");
        return;
    }

    double min_value = 0, max_value = 0;
    cv::minMaxLoc(depth, &min_value, &max_value);
    INFO("Depth min: %.3f m, max: %.3f m, mean: %.3f m", min_value, max_value, cv::mean(depth)[0]);

    float vmin = 0.0f, vmax = 1.0f;
    disparity_range(depth, vmin, vmax);

    cv::Mat spectral = colorize_depth(depth, true, vmin, vmax, DepthColorMap::Spectral);
    cv::Mat overlay  = depth_overlay(image, depth, vmin, vmax, 0.6f);

    // original | spectral depth | jet overlay
    cv::Mat result;
    cv::hconcat(std::vector<cv::Mat>{image, spectral, overlay}, result);
    cv::imwrite("Result-YoloDepth.jpg", result);
    cv::imwrite("Result-YoloDepth.tiff", depth);
    INFO("Save to Result-YoloDepth.jpg, Result-YoloDepth.tiff");
    engine.reset();
}

static void test_video(){

    auto engine = YoloDepth::create_infer(
        "yolo26s-depth.FP16.trtmodel",              // engine file
        0,                                          // gpu id
        false                                       // preprocess use multi stream
    );

    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    cv::VideoCapture cap("inference_depth/videos/davis_dolphins.mp4");
    // cv::VideoCapture cap(0);    // usb camera
    if(!cap.isOpened()){
        INFOE("Video open failed");
        return;
    }

    int width  = cap.get(cv::CAP_PROP_FRAME_WIDTH);
    int height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
    double fps = cap.get(cv::CAP_PROP_FPS);
    if(fps <= 1.0)
        fps = 25.0;

    cv::VideoWriter writer(
        "result_yolo_depth.mp4",
        cv::VideoWriter::fourcc('M', 'J', 'P', 'G'),
        fps, cv::Size(width * 2, height)
    );
    if(!writer.isOpened()){
        INFOE("VideoWriter open failed");
        return;
    }

    // keep the first frame's range so colors stay stable across the video
    bool has_range = false;
    float vmin = 0.0f, vmax = 1.0f;
    cv::Mat frame, result;
    while(cap.read(frame)){

        auto t0 = iLogger::timestamp_now_float();
        auto depth = engine->commit(frame).get();
        if(depth.empty())
            continue;

        if(!has_range){
            disparity_range(depth, vmin, vmax);
            has_range = true;
        }

        auto fee = iLogger::timestamp_now_float() - t0;
        INFO("fee = %.2f ms, FPS = %.2f", fee, 1 / fee * 1000);

        cv::Mat heat = colorize_depth(depth, true, vmin, vmax, DepthColorMap::Spectral);
        cv::hconcat(frame, heat, result);
        writer.write(result);
        cv::imshow("YoloDepth: Original | Depth", result);
        int key = cv::waitKey(1);
        if(key == 27)
            break;
    }

    INFO("Done.");
    cap.release();
    writer.release();
    cv::destroyAllWindows();
    engine.reset();
}

int app_yolo_depth(){

    test(TRT::Mode::FP16, "yolo26s-depth");
    // test_single_image();
    // test_video();
    return 0;
}
