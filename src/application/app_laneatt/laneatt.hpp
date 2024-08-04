#ifndef LANEATT_HPP
#define LANEATT_HPP

#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>

namespace LaneATT{

    const int N_OFFSETS = 72;
    const int N_STRIPS  = N_OFFSETS - 1;

    struct Lane{
        float unknow;
        float score;
        float start_y;
        float start_x;
        float length;
        float lane_xs[72];
        std::vector<cv::Point2f> points;
    };
    typedef std::vector<Lane> LaneArray;

    enum class NMSMethod : int{
        CPU = 0,
        FastGPU = 1
    };
    
    void image_to_tensor(const cv::Mat& image, std::shared_ptr<TRT::Tensor>& tensor, int ibatch);

    class Infer{
    public:
        virtual std::shared_future<LaneArray> commit(const cv::Mat& image) = 0;
        virtual std::vector<std::shared_future<LaneArray>> commits(const std::vector<cv::Mat>& images) = 0;
    };

    std::shared_ptr<Infer> create_infer(
        const std::string& engine_file, int gpuid, 
        float confidence_threshold = 0.5f, float nms_threshold = 50.f, 
        int nms_topk = 4, NMSMethod nms_method = NMSMethod::FastGPU, 
        int max_lanes = 256, bool use_multi_preprocess_stream = false
    );

};  // namespace LaneATT

#endif  // LANEATT_HPP
