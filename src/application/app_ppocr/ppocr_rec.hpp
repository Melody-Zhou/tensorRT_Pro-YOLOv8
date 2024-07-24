#ifndef PPOCR_REC_HPP
#define PPOCR_REC_HPP

#include <vector>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>

namespace PaddleOCR{
namespace SVTRRecognizer{

    struct Text{
        std::string text;
        float score;
        
        Text() = default;
        
        Text(std::string text, float score)
        :text(text), score(score){}
    };

    typedef std::vector<Text> TextArray;

    void image_to_tensor(const cv::Mat& image, std::shared_ptr<TRT::Tensor>& tensor, int ibatch);

    class TextRecognizer{
    public:
        virtual std::shared_future<Text> commit(const cv::Mat& image) = 0;
        virtual std::vector<std::shared_future<Text>> commits(const std::vector<cv::Mat>& images) = 0;
    };

    std::shared_ptr<TextRecognizer> create_recognizer(
        const std::string& engine_file, int gpuid, const std::string& dict_path = "ppocr/utils/ppocr_keys_v1.txt",
        int rec_batch_num = 4, bool use_multi_preprocess_stream = false
    );

};  // namespace SVTRRecognizer
};  // namespace PaddleOCR

#endif  // PPOCR_REC_HPP