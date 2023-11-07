#ifndef YOLO_POSE_HPP
#define YOLO_POSE_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>

namespace YoloPose{

    using namespace std;
    const int NUM_KEYPOINTS = 17;   // COCO Keypoins

    struct Box{
        float left, top, right, bottom, confidence;
        vector<cv::Point3f> keypoints;

        Box() = default;
        
        Box(float left, float top, float right, float bottom, float confidence)
        :left(left), top(top), right(right), bottom(bottom), confidence(confidence){
            keypoints.reserve(NUM_KEYPOINTS);
        }
    };
    typedef vector<Box> BoxArray;

    enum class NMSMethod : int{
        CPU = 0,         // General, for estimate mAP
        FastGPU = 1      // Fast NMS with a small loss of accuracy in corner cases
    };

    void image_to_tensor(const cv::Mat& image, shared_ptr<TRT::Tensor>& tensor, int ibatch);

    class Infer{
    public:
        virtual shared_future<BoxArray> commit(const cv::Mat& image) = 0;
        virtual vector<shared_future<BoxArray>> commits(const vector<cv::Mat>& images) = 0;
    };

    shared_ptr<Infer> create_infer(
        const string& engine_file, int gpuid,
        float confidence_threshold=0.25f, float nms_threshold=0.5f,
        NMSMethod nms_method = NMSMethod::FastGPU, int max_objects = 1024,
        bool use_multi_preprocess_stream = false
    );

}; // namespace YoloPose

#endif // YOLO_POSE_HPP