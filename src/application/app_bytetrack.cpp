
#include <builder/trt_builder.hpp>
#include <infer/trt_infer.hpp>
#include <common/ilogger.hpp>
#include <opencv2/opencv.hpp>
#include "app_yolo/yolo.hpp"
#include "app_bytetrack/byte_tracker.hpp"
#include "app_bytetrack/strack.hpp"
#include "tools/zmq_remote_show.hpp"

using namespace cv;
using namespace std;

bool requires(const char* name);
static const char* cocolabels[] = {
    "person", "bicycle", "car", "motorcycle", "airplane",
    "bus", "train", "truck", "boat", "traffic light", "fire hydrant",
    "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
    "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis",
    "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
    "skateboard", "surfboard", "tennis racket", "bottle", "wine glass",
    "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich",
    "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
    "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush"
};

static bool compile_models(){

    TRT::set_device(0);
    string model_file;

    const char* onnx_files[]{"yolov8s"};
    for(auto& name : onnx_files){
        if(not requires(name))
            return false;

        string onnx_file = iLogger::format("%s.onnx", name);
        string model_file = iLogger::format("%s.FP32.trtmodel", name);
        int test_batch_size = 1;
        
        if(not iLogger::exists(model_file)){
            bool ok = TRT::compile(
                TRT::Mode::FP32,            // FP32、FP16、INT8
                test_batch_size,            // max batch size
                onnx_file,                  // source
                model_file                  // saveto
            );

            if(!ok) return false;
        }
    }
    return true;
}

static void test_video(){
    
    TRT::set_device(0);
    INFO("===================== test yolov8s fp32 ==================================");

    if(!compile_models())
        return;

    auto detector = Yolo::create_infer("yolov8s.FP32.trtmodel", Yolo::Type::V8, 0, 0.5f);
    if(detector == nullptr){
        INFOE("Engine is nullptr");
        return;
    }
    // auto remote_show = create_zmq_remote_show("tcp://0.0.0.0:15556");
    // INFO("Use tools/show.py to remote show");
    
    VideoCapture cap("exp/test.mp4");
	if (!cap.isOpened()){
        INFOE("Could not open the video");
		return;
    }

	int frame_w = cap.get(CAP_PROP_FRAME_WIDTH);
	int frame_h = cap.get(CAP_PROP_FRAME_HEIGHT);
    int frame_fps = cap.get(CAP_PROP_FPS);
    long nFrame = static_cast<long>(cap.get(CAP_PROP_FRAME_COUNT));
    INFO("frame_w = %d, frame_h = %d, frame_fps = %d, Total frames: %ld", frame_w, frame_h, frame_fps, nFrame); 

    VideoWriter writer("demo.mp4", VideoWriter::fourcc('m', 'p', '4', 'v'), frame_fps, Size(frame_w, frame_h));

    ByteTrack::BYTETracker tracker(frame_fps, 30);

    Mat frame;

	while(cap.read(frame)){
        auto boxes = detector->commit(frame).get();
        auto output_stracks = tracker.update(boxes);

        for(auto& strack : output_stracks){
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(strack.class_label);
            auto tlwh = strack.tlwh;
            cv::rectangle(frame, cv::Rect(tlwh[0], tlwh[1], tlwh[2], tlwh[3]), cv::Scalar(b, g, r), 2);

            auto name = cocolabels[strack.class_label];
            auto caption = iLogger::format("id:%d %s %.2f", strack.track_id, name, strack.score);
            int width    = cv::getTextSize(caption, 0, 1, 2, nullptr).width + 10;
            cv::rectangle(frame, cv::Point(tlwh[0] - 3, tlwh[1] - 33), cv::Point(tlwh[0] + width, tlwh[1]), cv::Scalar(b, g, r), -1);
            cv::putText(frame, caption, cv::Point(tlwh[0], tlwh[1] - 5), 0, 1, cv::Scalar::all(0), 2, 16);
        }
        // remote_show->post(img);
        cv::imshow("frame", frame);
        int key = cv::waitKey(10);
        if(key == 27)
            break;
        // writer.write(frame);        
    }
    
    // char end_signal = 'x';
    // remote_show->post(&end_signal, 1);
    // INFO("save done.");
    cap.release();
    detector.reset();
    return;    
}

int app_bytetrack(){

    test_video();
    return 0;
}
