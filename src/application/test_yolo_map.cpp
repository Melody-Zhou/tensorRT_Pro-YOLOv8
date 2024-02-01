#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <common/json.hpp>
#include "app_yolo/yolo.hpp"
#include <vector>
#include <string>

using namespace std;

struct ImageItem{
    string image_file;
    Yolo::BoxArray detections;
};

vector<ImageItem> scan_dataset(const string& images_root){

    vector<ImageItem> output;
    auto image_files = iLogger::find_files(images_root, "*.jpg");

    for(int i = 0; i < image_files.size(); ++i){
        auto& image_file = image_files[i];

        if(!iLogger::exists(image_file)){
            INFOW("Not found: %s", image_file.c_str());
            continue;
        }

        ImageItem item;
        item.image_file = image_file;
        output.emplace_back(item);
    }
    return output;
}

static void inference(vector<ImageItem>& images, int deviceid, const string& engine_file, TRT::Mode mode, Yolo::Type type){

    auto engine = Yolo::create_infer(
        engine_file, type, deviceid, 0.03f, 0.65f,
        Yolo::NMSMethod::CPU, 10000
    );
    if(engine == nullptr){
        INFOE("Engine is nullptr");
        return;
    }

    int nimages = images.size();
    vector<shared_future<Yolo::BoxArray>> image_results(nimages);
    for(int i = 0; i < nimages; ++i){
        if(i % 100 == 0){
            INFO("Commit %d / %d", i+1, nimages);
        }
        image_results[i] = engine->commit(cv::imread(images[i].image_file));
    }
    
    for(int i = 0; i < nimages; ++i)
        images[i].detections = image_results[i].get();
}

bool save_to_json(const vector<ImageItem>& images, const string& file){

    Json::Value predictions(Json::arrayValue);
    for(int i = 0; i < images.size(); ++i){
        auto& image = images[i];
        auto file_name = iLogger::file_name(image.image_file, false);
        string image_id = file_name;

        auto& boxes = image.detections;
        for(auto& box : boxes){
            Json::Value jitem;
            jitem["image_id"] = image_id;
            jitem["category_id"] = box.class_label;
            jitem["score"] = box.confidence;

            auto& bbox = jitem["bbox"];
            bbox.append(box.left);
            bbox.append(box.top);
            bbox.append(box.right - box.left);
            bbox.append(box.bottom - box.top);
            predictions.append(jitem);
        }
    }
    return iLogger::save_file(file, predictions.toStyledString());
}

int test_yolo_map(){
    
    /*
    结论：
    1. YoloV5在tensorRT下和pytorch下，只要输入一样，输出的差距最大值是1e-3
    2. YoloV5-6.0的mAP，官方代码跑下来是mAP@.5:.95 = 0.367, mAP@.5 = 0.554，与官方声称的有差距
    3. 这里的tensorRT版本测试的精度为：mAP@.5:.95 = 0.357, mAP@.5 = 0.539，与pytorch结果有差距
    4. cv2.imread与cv::imread，在操作jpeg图像时，在我这里测试读出的图像值不同，最大差距有19。而png图像不会有这个问题
        若想完全一致，请用png图像
    5. 预处理部分，若采用letterbox的方式做预处理，由于tensorRT这里是固定640x640大小，测试采用letterbox并把多余部分
        设置为0. 其推理结果与pytorch相近，但是依旧有差别
    6. 采用warpAffine和letterbox两种方式的预处理结果，在mAP上没有太大变化（小数点后三位差）
    7. mAP差一个点的原因可能在固定分辨率这件事上，还有是pytorch实现的所有细节并非完全加入进来。这些细节可能有没有
        找到的部分
    8. 默认设置的 conf_thresh=0.03f, nms_thresh=0.65f
    */

    auto images = scan_dataset("/home/jarvis/Learn/Datasets/VOC_PTQ/images/val");
    INFO("images.size = %d", images.size());

    string model = "best.minmax.INT8.trtmodel";
    inference(images, 0, model, TRT::Mode::INT8, Yolo::Type::V6);
    save_to_json(images, model + ".prediction.json");
    return 0;
}
