#include "utils.hpp"
#include "cuosd/cuosd.h"
#include "cuosd/gpu_image.h"
#include <common/cuda_tools.hpp>
#include <locale>
#include <codecvt>

namespace PaddleOCR{

    using namespace cv;
    using namespace std;

    cv::Mat get_rotate_crop_image(const cv::Mat& src_image, const vector<vector<int>>& box){

        std::array<cv::Point, 4> points;
        for(int i = 0; i < 4; ++i){
            points[i] = cv::Point(box[i][0], box[i][1]);
        }

        auto [minPoint, maxPoint] = std::minmax_element(points.begin(), points.end(), [](const cv::Point& a, const cv::Point& b){
            return a.x < b.x;
        });
        int left  = minPoint->x;
        int right = maxPoint->x;
        minPoint = std::min_element(points.begin(), points.end(), [](const cv::Point& a, const cv::Point& b){
            return a.y < b.y;
        });
        maxPoint = std::max_element(points.begin(), points.end(), [](const cv::Point& a, const cv::Point& b){
            return a.y < b.y;
        });
        int top    = minPoint->y;
        int bottom = maxPoint->y;
        
        cv::Rect roi(left, top, right - left, bottom - top);
        cv::Mat img_crop = src_image(roi);
        int width  = cv::norm(points[0] - points[1]);
        int height = cv::norm(points[0] - points[3]);

        std::array<cv::Point2f, 4> srcPoints = {
            cv::Point2f(points[0].x - left, points[0].y - top),
            cv::Point2f(points[1].x - left, points[1].y - top),
            cv::Point2f(points[2].x - left, points[2].y - top),
            cv::Point2f(points[3].x - left, points[3].y - top)
        };        
        std::array<cv::Point2f, 4> dstPoints = {
            cv::Point2f(0.0f, 0.0f),
            cv::Point2f(width, 0.0f),
            cv::Point2f(width, height),
            cv::Point2f(0.0f, height)
        };

        cv::Mat M = cv::getPerspectiveTransform(srcPoints.data(), dstPoints.data());
        cv::Mat dst_img;
        cv::warpPerspective(img_crop, dst_img, M, cv::Size(width, height), cv::BORDER_REPLICATE);

        if(float(dst_img.rows) >= float(dst_img.cols) * 1.5){
            cv::Mat rotated;
            cv::transpose(dst_img, rotated);
            cv::flip(rotated, rotated, 0);
            return rotated;
        }else{
            return dst_img;
        }
    } 

    struct Polyline
    {
        int* h_pts      = nullptr;
        int* d_pts      = nullptr;
        int n_pts       = 0;
    };

    static Polyline* create_polyline(vector<vector<int>>& box, int width){
        Polyline* output = new Polyline();
        std::vector<gpu::Point> points;
        for(auto& point : box){
            points.push_back(gpu::Point({point[0] + width, point[1]}));
        }

        output->n_pts = points.size();
        output->h_pts = (int *)malloc(output->n_pts * 2 * sizeof(int));
        memcpy(output->h_pts, points.data(), output->n_pts * 2 * sizeof(int));
        checkCudaRuntime(cudaMalloc(&output->d_pts, output->n_pts * 2 * sizeof(int)));
        checkCudaRuntime(cudaMemcpy(output->d_pts, points.data(), output->n_pts * 2 * sizeof(int), cudaMemcpyHostToDevice));
        return output;
    }

    void draw_ocr_box_txt(cv::Mat& image, DBDetector::BoxArray& boxes, vector<string>& texts, string save_path){

        cudaStream_t stream = nullptr;
        checkCudaRuntime(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        auto context = cuosd_context_create();
        cv::cvtColor(image, image, cv::COLOR_BGR2RGB);
        gpu::Image* gpu_image = gpu::create_image(image.cols, image.rows, gpu::ImageFormat::RGB);
        checkCudaRuntime(cudaMemcpy(gpu_image->data0, image.data, gpu_image->stride * gpu_image->height, cudaMemcpyHostToDevice));
        
        const char* font_path = "ppocr/font/simfang.ttf";
        for(int i = 0; i < texts.size(); ++i){
            uint8_t b, g, r;
            tie(b, g, r) = iLogger::random_color(i);
            auto& box = boxes[i];
            int x = box[0][0] + image.cols / 2;
            int y = box[0][1];
            string text = texts[i];
            auto distance = [](const vector<int>& a, const vector<int>& b){
                int dx = a[0] - b[0];
                int dy = a[1] - b[1];
                return int(std::sqrt(dx * dx + dy * dy));
            };
            int box_width  = distance(box[0], box[1]);
            int box_height = distance(box[0], box[3]);
            if(box_height > 2 * box_width && box_height > 30){
                wstring_convert<std::codecvt_utf8<wchar_t>> conv;
                wstring wtext = conv.from_bytes(text);
                for(int i = 0; i < wtext.size(); ++i){
                    std::string utf8char = conv.to_bytes(wtext[i]);
                    cuosd_draw_text(context, utf8char.c_str(), 2, font_path, x, y+10*i, {0, 0, 0, 255}, {255, 255, 255, 0});
                }
            }else{
                cuosd_draw_text(context, text.c_str(), 2, font_path, x, y, {0, 0, 0, 255}, {255, 255, 255, 0});
            }

            Polyline* polyline = create_polyline(box, image.cols / 2);
            cuosd_draw_polyline(context, polyline->h_pts, polyline->d_pts, polyline->n_pts, 1, true, {r, g, b, 255});
        }

        cuosd_apply(context, gpu_image->data0, gpu_image->data1, gpu_image->width, gpu_image->stride, gpu_image->height, cuOSDImageFormat::RGB, stream, true);
        cuosd_context_destroy(context);
        gpu::save_image(gpu_image, save_path.c_str(), stream);
        checkCudaRuntime(cudaStreamDestroy(stream));
    }   
};
