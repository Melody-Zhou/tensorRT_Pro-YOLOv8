#include <common/ilogger.hpp>
#include <builder/trt_builder.hpp>
#include "app_ppocr/utils.hpp"
#include "app_ppocr/ppocr.hpp"
#include "app_ppocr/ppocr_det.hpp"
#include "app_ppocr/ppocr_cls.hpp"
#include "app_ppocr/ppocr_rec.hpp"

using namespace std;
using namespace PaddleOCR;

static void model_compile(TRT::Mode mode, const string& model){

    int deviceid = 0;
    auto mode_name = TRT::mode_string(mode);
    TRT::set_device(deviceid);

    auto int8process = [=](int current, int count, const vector<string>& files, shared_ptr<TRT::Tensor>& tensor){

        INFO("Int8 %d / %d", current, count);

        for(int i = 0; i < files.size(); ++i){
            auto image = cv::imread(files[i]);
            DBDetector::image_to_tensor(image, tensor, i);      // det
            // Classifier::image_to_tensor(image, tensor, i);      // cls
            // SVTRRecognizer::image_to_tensor(image, tensor, i);  // rec
        }
    };

    const char* name = model.c_str();
    INFO("===================== compile %s %s ==================================", mode_name, name);

    string onnx_file = iLogger::format("%s.onnx", name);
    string model_file = iLogger::format("%s.%s.trtmodel", name, mode_name);
    int test_batch_size = 8;

    if(not iLogger::exists(model_file)){
        TRT::compile(
            mode,                       // FP32、FP16、INT8
            test_batch_size,            // max batch size
            onnx_file,                  // source 
            model_file,                 // save to
            {},
            int8process
        );        
    }
    INFO("engine build done. save to %s", model_file.c_str());
}

static void test_ocr(){

    auto image = cv::imread("ppocr/imgs/lite_demo.png");
    auto text_detector   = DBDetector::create_detector("ppocr_det.sim.FP16.trtmodel", 0);
    auto text_classifier = Classifier::create_classifier("ppocr_cls.sim.FP16.trtmodel", 0);
    auto text_recognizer = SVTRRecognizer::create_recognizer("ppocr_rec.sim.FP16.trtmodel", 0);

    auto box_array = text_detector->commit(image).get();
    vector<cv::Mat> img_list;
    for(auto& box : box_array){
        cv::Mat crop_image;
        crop_image = get_rotate_crop_image(image, box);
        // auto name = iLogger::format("crops/crop_image_%d.jpg", i);
        // cv::imwrite(name, crop_image);        
        img_list.push_back(crop_image);
    }

    auto angle_array = text_classifier->commits(img_list);
    angle_array.back().get();
    for(int i = 0; i < angle_array.size(); ++i){
        auto angle = angle_array[i].get();
        if(angle == 1){
            cv::rotate(img_list[i], img_list[i], 1);
        }
    }

    auto text_array =  text_recognizer->commits(img_list);
    text_array.back().get();

    // visualize
    DBDetector::BoxArray boxes;
    boxes.reserve(box_array.size());
    vector<string> texts;
    texts.reserve(text_array.size());
    cv::Mat left  = image;
    cv::Mat right = cv::Mat::zeros(left.size(), left.type());
    right.setTo(cv::Scalar(255, 255, 255));
    cv::Mat image_show;
    cv::hconcat(left, right, image_show);    
    for(int i = 0; i < text_array.size(); ++i){
        uint8_t b, g, r;
        tie(b, g, r) = iLogger::random_color(i);
        auto& box   = box_array[i];
        auto text   = text_array[i].get();
        string str  = text.text;
        float score = text.score;
        if(score < 0.5)
            continue;
        str.erase(std::remove(str.begin(), str.end(), '\r'), str.end());
        texts.push_back(str);
        boxes.push_back(box);

        // cv::Point points[4];
        // for(int i = 0; i < box.size(); ++i){
        //     points[i] = cv::Point(int(box[i][0]), int(box[i][1]));
        // }
        // const cv::Point* ppt[1] = {points};
        // int npt[] = {4};
        // cv::polylines(right, ppt, npt, 1, true, cv::Scalar(b, g, r), 1);         
    }
    draw_ocr_box_txt(image_show, boxes, texts, "result_ocr.jpg");
    text_detector.reset();
    text_classifier.reset();
    text_recognizer.reset();    
}

static void test_ocr_system(){

    auto image = cv::imread("ppocr/imgs/lite_demo.png");
    OcrParameter param;
    param.det_engine_file = "ppocr_det.sim.FP16.trtmodel";
    param.cls_engine_file = "ppocr_cls.sim.FP16.trtmodel";
    param.rec_engine_file = "ppocr_rec.sim.FP16.trtmodel";
    auto text_system = PaddleOCR::create_system(param);
    auto ocr_result  = text_system->commit(image);
    if(ocr_result.boxes.empty())
        return;
    
    // visualize
    auto& boxes = ocr_result.boxes;
    auto& texts = ocr_result.texts;
    DBDetector::BoxArray box_array;
    box_array.reserve(boxes.size());
    vector<string> text_array;
    text_array.reserve(texts.size());
    for(int i = 0; i < texts.size(); ++i){
        auto& box   = boxes[i];
        auto& text  = texts[i];
        string str  = text.text;
        float score = text.score;
        if(score < 0.5)
            continue;
        str.erase(std::remove(str.begin(), str.end(), '\r'), str.end());
        text_array.push_back(str);
        box_array.push_back(box);
    }

    cv::Mat left  = image;
    cv::Mat right = cv::Mat::zeros(left.size(), left.type()); 
    right.setTo(cv::Scalar(255, 255, 255));
    cv::Mat image_show;
    cv::hconcat(left, right, image_show);
    draw_ocr_box_txt(image_show, box_array, text_array, "result_ocr.jpg");
    INFO("save to result_ocr.jpg, text box number = %d", box_array.size()); 
    text_system.reset();
}

static void test_ocr_system_images(){

    auto files = iLogger::find_files("ppocr/imgs", "*.jpg;*.jpeg;*.png;*.gif;*.tif");
    vector<cv::Mat> images;
    for(int i = 0; i < files.size(); ++i){
        auto image = cv::imread(files[i]);
        images.emplace_back(image);
    }

    OcrParameter param;
    param.det_engine_file = "ppocr_det.sim.FP16.trtmodel";
    param.cls_engine_file = "ppocr_cls.sim.FP16.trtmodel";
    param.rec_engine_file = "ppocr_rec.sim.FP16.trtmodel";
    auto text_system = PaddleOCR::create_system(param);
    auto ocr_result_array = text_system->commits(images);
    string root = iLogger::format("ppocrv4_result");
    iLogger::rmtree(root);
    iLogger::mkdir(root);

    for(int i = 0; i < ocr_result_array.size(); ++i){
        auto& image = images[i];
        auto ocr_result = ocr_result_array[i];
        if(ocr_result.boxes.empty())
            continue;
        
        // visualize
        auto& boxes = ocr_result.boxes;
        auto& texts = ocr_result.texts;
        DBDetector::BoxArray box_array;
        box_array.reserve(boxes.size());
        vector<string> text_array;
        text_array.reserve(texts.size());
        for(int i = 0; i < texts.size(); ++i){
            auto& box   = boxes[i];
            auto& text  = texts[i];
            string str  = text.text;
            float score = text.score;
            if(score < 0.5)
                continue;
            str.erase(std::remove(str.begin(), str.end(), '\r'), str.end());
            text_array.push_back(str);
            box_array.push_back(box);
        }

        cv::Mat left  = image;
        cv::Mat right = cv::Mat::zeros(left.size(), left.type()); 
        right.setTo(cv::Scalar(255, 255, 255));
        cv::Mat image_show;
        cv::hconcat(left, right, image_show);
        string file_name = iLogger::file_name(files[i], false);
        string save_path = iLogger::format("%s/%s.jpg", root.c_str(), file_name.c_str());
        draw_ocr_box_txt(image_show, box_array, text_array, save_path); 
        INFO("save to %s, text box number = %d", save_path.c_str(), box_array.size());
    }
    text_system.reset();
}

int app_ppocr(){

    // test_ocr();
    test_ocr_system();
    // test_ocr_system_images();
    // model_compile(TRT::Mode::FP32, "ppocr_det.sim");

    return 0;
}