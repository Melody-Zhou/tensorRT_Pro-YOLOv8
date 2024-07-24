#ifndef PPOCR_DET_HPP
#define PPOCR_DET_HPP

#include <vector>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>

namespace PaddleOCR{
namespace DBDetector{

    typedef std::vector<std::vector<std::vector<int>>> BoxArray;

    void image_to_tensor(const cv::Mat& image, std::shared_ptr<TRT::Tensor>& tensor, int ibatch);

    class TextDetector{
    public:
        virtual std::shared_future<BoxArray> commit(const cv::Mat& image) = 0;
    };

    std::shared_ptr<TextDetector> create_detector(
        const std::string& engine_file, int gpuid,
        float mask_thresh = 0.3f, float box_thresh = 0.6f,
        float unclip_ratio = 1.5f, int min_size = 3,
        int max_candidates = 1000, bool use_multi_preprocess_stream = false
    );

};  // namespace DBDetector    
};  // namespace PaddleOCR

#endif  // PPOCR_DET_HPP