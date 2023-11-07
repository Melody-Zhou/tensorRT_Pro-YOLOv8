#ifndef YOLO_CLS_HPP
#define YOLO_CLS_HPP

#include <vector>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>

namespace YoloCls{

    using namespace std;

    struct Prob{
        int class_label;
        float confidence;

        Prob() = default;

        Prob(int class_label, float confidence)
        :class_label(class_label), confidence(confidence){}
    };
    typedef std::vector<Prob> ProbArray;

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch);

    class Infer{
    public:
        virtual shared_future<ProbArray> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<ProbArray>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, int gpuid,
        bool use_multi_preprocess_stream = false
    );

}; // namespace YoloCls

#endif // YOLO_CLS_HPP