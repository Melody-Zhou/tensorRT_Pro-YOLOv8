#ifndef PPOCR_HPP
#define PPOCR_HPP

#include <vector>
#include <future>
#include <opencv2/opencv.hpp>
#include "ppocr_det.hpp"
#include "ppocr_cls.hpp"
#include "ppocr_rec.hpp"

namespace PaddleOCR{
    
    struct OcrParameter{
        std::string det_engine_file;
        std::string cls_engine_file;
        std::string rec_engine_file;
        int gpuid = 0;
        bool use_multi_preprocess_stream = false; 
        // detector param
        float mask_thresh  = 0.3f;
        float box_thresh   = 0.6f;
        float unclip_ratio = 1.5f;
        int min_size       = 3;
        int max_candidates = 1000;
        // classifier param
        bool use_angle_cls = false;
        float cls_thresh   = 0.9f;
        int cls_batch_num  = 4;
        // recognizer param
        std::string dict_path = "ppocr/utils/ppocr_keys_v1.txt";
        int rec_batch_num     = 4;
    };

    struct OcrResult{
        DBDetector::BoxArray boxes;
        SVTRRecognizer::TextArray texts;

        OcrResult() = default;

        OcrResult(DBDetector::BoxArray boxes, SVTRRecognizer::TextArray texts)
        :boxes(boxes), texts(texts){}
    };

    typedef std::vector<OcrResult> OcrResultArray;

    class TextSystem{
    public:

        virtual OcrResult commit(const cv::Mat& image) = 0;
        virtual OcrResultArray commits(const std::vector<cv::Mat>& images) = 0;
    };

    std::shared_ptr<TextSystem> create_system(const OcrParameter& param);

};  // namespace PaddleOCR
#endif  // PPOCR_HPP