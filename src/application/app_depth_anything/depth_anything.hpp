#ifndef DEPTH_ANYTHING_HPP
#define DEPTH_ANYTHING_HPP

#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>

namespace DepthAnything{

    using namespace std;

    enum class Type : int{
        V1 = 0,
        V2 = 1
    };

    enum class InterpolationDevice : int{
        CPU = 0,
        FastGPU = 1
    };

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch);
    
    class Infer{
    public:
        virtual shared_future<cv::Mat> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<cv::Mat>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, Type type, int gpuid,
        InterpolationDevice interpolation_device = InterpolationDevice::CPU,
        bool use_multi_preprocess_stream = false
    );
    const char* type_name(Type type);

};  // namespace DepthAnything

#endif  // DEPTH_ANYTHING_HPP