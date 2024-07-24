#include "ppocr.hpp"
#include "utils.hpp"
#include <common/ilogger.hpp>

namespace PaddleOCR{

    using namespace cv;
    using namespace std;
    
    class TextSystemImpl : public TextSystem{
    public:
        virtual bool startup(const OcrParameter& param){
            text_detector_ = DBDetector::create_detector(
                param.det_engine_file, param.gpuid, param.mask_thresh,
                param.box_thresh, param.unclip_ratio, param.min_size,
                param.max_candidates, param.use_multi_preprocess_stream
            );
            if(text_detector_ == nullptr){
                INFOE("Failed to create detector.");
                return false;
            }

            if(param.use_angle_cls){
                text_classifier_ = Classifier::create_classifier(
                    param.cls_engine_file, param.gpuid, param.cls_thresh,
                    param.cls_batch_num, param.use_multi_preprocess_stream 
                );
                if(text_classifier_ == nullptr){
                    INFOE("Failed to create classifier.");
                    return false;
                }
            }else{
                text_classifier_ = nullptr;
            }

            text_recognizer_ = SVTRRecognizer::create_recognizer(
                param.rec_engine_file, param.gpuid, param.dict_path,
                param.rec_batch_num, param.use_multi_preprocess_stream
            );
            if(text_recognizer_ == nullptr){
                INFOE("Failed to create recognizer.");
                return false;
            }

            return true;
        }

        virtual OcrResult commit(const Mat& image) override{

            OcrResult ocr_reuslt;
            auto box_array = text_detector_->commit(image).get();
            if(box_array.empty()){
                INFOW("There is no text detected.");
                return ocr_reuslt;
            }
            ocr_reuslt.boxes = box_array;
            vector<cv::Mat> img_list;
            for(auto& box : box_array){
                cv::Mat crop_image;
                crop_image = get_rotate_crop_image(image, box);
                img_list.push_back(crop_image);
            }

            if(text_classifier_ != nullptr){
                auto angle_array = text_classifier_->commits(img_list);
                angle_array.back().get();
                for(int i = 0; i < angle_array.size(); ++i){
                    auto angle = angle_array[i].get();
                    if(angle == 1){
                        cv::rotate(img_list[i], img_list[i], 1);
                    }
                }
            }

            auto text_array = text_recognizer_->commits(img_list);
            text_array.back().get();
            for(int i = 0; i < text_array.size(); ++i){
                auto text = text_array[i].get();
                ocr_reuslt.texts.push_back(text);
            }

            return ocr_reuslt;
        }

        virtual OcrResultArray commits(const vector<Mat>& images) override{

            OcrResultArray ocr_result_array;
            for(auto& image : images){
                auto ocr_result = commit(image);
                ocr_result_array.push_back(ocr_result);
            }
            return ocr_result_array;
        }

    private:

        shared_ptr<DBDetector::TextDetector> text_detector_;
        shared_ptr<Classifier::TextClassifier> text_classifier_;
        shared_ptr<SVTRRecognizer::TextRecognizer> text_recognizer_;

    };

    shared_ptr<TextSystem> create_system(const OcrParameter& param){
        shared_ptr<TextSystemImpl> instance(new TextSystemImpl());
        if(!instance->startup(param)){
            instance.reset();
        }
        return instance;
    }
};