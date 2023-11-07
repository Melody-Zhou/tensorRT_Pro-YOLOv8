#ifndef YOLO_SEG_HPP
#define YOLO_SEG_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>
#include <common/object_detector.hpp>

namespace YoloSeg{

    using namespace std;
    using namespace ObjectDetector;

    struct InstanceSegmentMap {
    int width = 0, height = 0;      // width % 8 == 0
    int left = 0, top = 0;          // 160x160 feature map
    unsigned char *data = nullptr;  // is width * height memory

    InstanceSegmentMap(int width, int height);
    virtual ~InstanceSegmentMap();
    };

    struct Box {
    float left, top, right, bottom, confidence;
    int class_label;
    std::shared_ptr<InstanceSegmentMap> seg;  // mask

    Box() = default;
    Box(float left, float top, float right, float bottom, float confidence, int class_label)
        : left(left),
            top(top),
            right(right),
            bottom(bottom),
            confidence(confidence),
            class_label(class_label) {}
    };
    typedef std::vector<Box> BoxArray;

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

}; // namespace YoloSeg

#endif // YOLO_SEG_HPP