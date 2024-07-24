#ifndef UTILS_HPP
#define UTILS_HPP

#include "app_ppocr/ppocr_det.hpp"
#include "app_ppocr/ppocr_rec.hpp"
#include <opencv2/opencv.hpp>

namespace PaddleOCR{

    cv::Mat get_rotate_crop_image(const cv::Mat& src_image, const std::vector<std::vector<int>>& box);

    void draw_ocr_box_txt(cv::Mat& image,  DBDetector::BoxArray& boxes, std::vector<std::string>& texts, std::string save_path);

};  // namespace PaddleOCR

#endif  // UTILS_HPP