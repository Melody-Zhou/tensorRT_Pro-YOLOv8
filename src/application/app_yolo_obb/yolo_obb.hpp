#ifndef YOLO_OBB_HPP
#define YOLO_OBB_HPP

#include <vector>
#include <memory>
#include <string>
#include <future>
#include <opencv2/opencv.hpp>
#include <common/trt_tensor.hpp>

namespace YoloOBB{

    using namespace std;

    struct Box{
        float center_x, center_y, width, height, angle, confidence;
        int class_label;

        Box() = default;

        Box(float center_x, float center_y, float width, float height, float angle, float confidence, int class_label)
        :center_x(center_x), center_y(center_y), width(width), height(height), angle(angle), confidence(confidence), class_label(class_label){}
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

}; // namespace YoloOBB

#endif // YOLO_OBB_HPP