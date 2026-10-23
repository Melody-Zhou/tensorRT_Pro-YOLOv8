#ifndef YOLO_DEPTH_HPP
#define YOLO_DEPTH_HPP

#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>

namespace YoloDepth{

    using namespace std;

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch);

    class Infer{
    public:
        virtual shared_future<cv::Mat> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<cv::Mat>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, int gpuid,
        bool use_multi_preprocess_stream = false
    );

}; // namespace YoloDepth

#endif // YOLO_DEPTH_HPP
