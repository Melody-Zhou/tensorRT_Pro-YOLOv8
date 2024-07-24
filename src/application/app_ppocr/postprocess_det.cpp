#include "ppocr_det.hpp"
#include "clipper.hpp"

namespace PaddleOCR{
namespace DBDetector{
    
    using namespace cv;
    using namespace std;
    using namespace ClipperLib;

    static vector<vector<float>> mat_to_vector(const Mat& mat){
        vector<vector<float>> img_vec;
        img_vec.reserve(mat.rows);
        
        for(int i = 0; i < mat.rows; ++i){
            vector<float> tmp;
            tmp.reserve(mat.cols);

            for(int j = 0; j < mat.cols; ++j){
                tmp.push_back(mat.at<float>(i, j));
            }
            img_vec.push_back(std::move(tmp));
        }
        return img_vec;
    }

    static tuple<vector<vector<float>>, float> get_mini_boxes(const cv::RotatedRect& box){
        float sside = std::min(box.size.width, box.size.height);
        
        cv::Mat points;
        cv::boxPoints(box, points);

        auto array = mat_to_vector(points);
        std::sort(array.begin(), array.end(), [](const vector<float>& a, const vector<float>& b){
            if(a[0] != b[0])
                return a[0] < b[0];
            return false;
        });

        vector<float> idx1 = array[0], idx2 = array[1],
                      idx3 = array[2], idx4 = array[3];
        if(array[3][1] <= array[2][1]){
            idx2 = array[3];
            idx3 = array[2];
        }else{
            idx2 = array[2];
            idx3 = array[3];
        }

        if(array[1][1] <= array[0][1]){
            idx1 = array[1];
            idx4 = array[0];
        }else{
            idx1 = array[0];
            idx4 = array[1];
        }

        array[0] = idx1;
        array[1] = idx2;
        array[2] = idx3;
        array[3] = idx4;

        return make_tuple(array, sside);
    }

    template<typename T>
    T clamp(T val, T minVal, T maxVal) {
        return (val < minVal) ? minVal : (val > maxVal) ? maxVal : val;
    }

    static float box_score(const Mat& pred, const vector<vector<float>>& box_array){
        int width  = pred.cols;
        int height = pred.rows;

        float box_x[4] = {box_array[0][0], box_array[1][0], box_array[2][0], box_array[3][0]};
        float box_y[4] = {box_array[0][1], box_array[1][1], box_array[2][1], box_array[3][1]};

        int xmin = clamp(int(std::floor(*(std::min_element(box_x, box_x + 4)))), 0, width - 1);
        int xmax = clamp(int(std::ceil(*(std::max_element(box_x, box_x + 4)))), 0, width - 1);
        int ymin = clamp(int(std::floor(*(std::min_element(box_y, box_y + 4)))), 0, height - 1);
        int ymax = clamp(int(std::ceil(*(std::max_element(box_y, box_y + 4)))), 0, height - 1);

        Mat mask = cv::Mat::zeros(ymax - ymin + 1, xmax - xmin + 1, CV_8UC1);
        cv::Point root_point[4];
        root_point[0] = cv::Point(int(box_x[0]) - xmin, int(box_y[0]) - ymin);
        root_point[1] = cv::Point(int(box_x[1]) - xmin, int(box_y[1]) - ymin);
        root_point[2] = cv::Point(int(box_x[2]) - xmin, int(box_y[2]) - ymin);
        root_point[3] = cv::Point(int(box_x[3]) - xmin, int(box_y[3]) - ymin);

        const cv::Point* ppt[1] = {root_point};
        int npt[] = {4};
        cv::fillPoly(mask, ppt, npt, 1, cv::Scalar(1));
        
        Mat croppedImg;
        pred(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1)).copyTo(croppedImg);

        return cv::mean(croppedImg, mask)[0];
    }

    static float get_contour_area(const vector<vector<float>>& box, float unclip_ratio){
        float distance = 1.0;
        const int pts_num = 4;
        float area = 0.0f;
        float dist = 0.0f;
        for(int i = 0; i < pts_num; ++i){
            int next = (i + 1) % pts_num;
            float dx = box[i][0] - box[next][0];
            float dy = box[i][1] - box[next][1];

            area += box[i][0] * box[next][1] - box[i][1] * box[next][0];
            dist += std::sqrt(dx * dx + dy * dy);        
        }

        area = std::abs(area / 2.0f);
        if(dist != 0.0f){
            distance = area * unclip_ratio / dist;
        }
        return distance;
    }

    static cv::RotatedRect box_unclip(const vector<vector<float>>& box, float unclip_ratio){
        float distance = get_contour_area(box, unclip_ratio);
        ClipperOffset offset;
        Path p;
        p << IntPoint(int(box[0][0]), int(box[0][1]))
          << IntPoint(int(box[1][0]), int(box[1][1]))
          << IntPoint(int(box[2][0]), int(box[2][1]))
          << IntPoint(int(box[3][0]), int(box[3][1]));
        offset.AddPath(p, jtRound, etClosedPolygon);

        Paths soln;
        offset.Execute(soln, distance);
        vector<cv::Point2f> points;
        
        for(int i = 0; i < soln.size(); ++i){
            for(int j = 0; j < soln[soln.size() - 1].size(); ++j){
                points.emplace_back(soln[i][j].X, soln[i][j].Y);
            }
        }
        cv::RotatedRect res;
        if(points.size() <= 0){
            res = cv::RotatedRect(cv::Point2f(0, 0), cv::Size2f(1, 1), 0);
        }else{
            res = cv::minAreaRect(points);
        }
        return res;
    }

    static void boxes_from_bitmap(
        const Mat& pred, const Mat& bitmap, vector<vector<vector<int>>>& box_array, 
        float box_thresh, float unclip_ratio, int min_size, int max_candidates
    ){
        
        int width  = bitmap.cols;
        int height = bitmap.rows;
        
        vector<vector<cv::Point>> contours;
        vector<cv::Vec4i> hierarchy;

        cv::findContours(bitmap, contours, hierarchy, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
        int num_contours = contours.size() >= max_candidates ? max_candidates : contours.size();

        Mat contour_image;
        cv::cvtColor(bitmap, contour_image, cv::COLOR_GRAY2BGR);
        
        // for(auto& contour : contours){
        //     vector<vector<cv::Point>> single_contour = {contour};
        //     cv::drawContours(contour_image, single_contour, -1, cv::Scalar(0, 0, 255), 2);
        // }
        // cv::imwrite("contour_image.jpg", contour_image);

        vector<vector<vector<int>>> boxes;
        for(auto& contour : contours){
            if(contour.size() <= min_size)
                continue;
            vector<vector<float>> array;
            float sside;
            auto box = cv::minAreaRect(contour);
            tie(array, sside) = get_mini_boxes(box);
            if(sside < min_size)
                continue;
            
            float score = box_score(pred, array);
            if(score < box_thresh)
                continue;
            auto points = box_unclip(array, unclip_ratio);
            // difference
            if(points.size.height < 1.001 & points.size.width < 1.001)
                continue;
            
            vector<vector<float>> cliparray;
            tie(cliparray, sside) = get_mini_boxes(points);
            if(sside < min_size + 2)
                continue;
            
            int dest_width  = pred.cols;
            int dest_height = pred.rows;

            vector<vector<int>> intcliparray;
            intcliparray.reserve(4);
            float x_scale = float(dest_width)  / float(width);
            float y_scale = float(dest_height) / float(height);
            
            for(int i = 0; i < 4; ++i){
                int x = int(clamp(std::roundf(cliparray[i][0] * x_scale), 0.0f, float(dest_width)));
                int y = int(clamp(std::roundf(cliparray[i][1] * y_scale), 0.0f, float(dest_height)));
                intcliparray.push_back({x, y});
            }

            box_array.emplace_back(intcliparray);
        }
    }

    static vector<vector<int>> order_points_clockwise(vector<vector<int>>& box){
        std::sort(box.begin(), box.end(), [](const vector<int>& a, const vector<int>& b){
            if(a[0] != b[0])
                return a[0] < b[0];
            return false;
        });

        vector<vector<int>> leftmost  = {box[0], box[1]};
        vector<vector<int>> rightmost = {box[2], box[3]};

        if(leftmost[0][1] > leftmost[1][1]){
            std::swap(leftmost[0], leftmost[1]);
        }
        
        if(rightmost[0][1] > rightmost[1][1]){
            std::swap(rightmost[0], rightmost[1]);
        }

        vector<vector<int>> rect = {leftmost[0], rightmost[0], rightmost[1], leftmost[1]};
        return rect;
    }

    static vector<vector<vector<int>>> filter_boxes(vector<vector<vector<int>>>& boxes, int src_h, int src_w, float ratio_h, float ratio_w){
        
        vector<vector<vector<int>>> boxes_filter;
        
        for(auto& box : boxes){
            box = order_points_clockwise(box);
            for(int i = 0; i < box.size(); ++i){
                box[i][0] /= ratio_w;
                box[i][1] /= ratio_h;
                box[i][0] = int(std::min(std::max(box[i][0], 0), src_w - 1));
                box[i][1] = int(std::min(std::max(box[i][1], 0), src_h - 1));
            }
            int rec_width  = int(sqrt(pow(box[0][0] - box[1][0], 2) + pow(box[0][1] - box[1][1], 2)));
            int rec_height = int(sqrt(pow(box[0][0] - box[3][0], 2) + pow(box[0][1] - box[3][1], 2)));
            if(rec_width <= 3 || rec_height <= 3)
                continue;
            boxes_filter.push_back(box);
        }

        return boxes_filter;
    }

    void detector_postprocess(
        const Mat& pred_map, BoxArray& boxes, int src_h, int src_w, int dst_h, int dst_w,
        float mask_thresh, float box_thresh, float unclip_ratio, int min_size, int max_candidates
    ){
        Mat cbuf_map;
        cv::convertScaleAbs(pred_map, cbuf_map, 255.0);
        Mat bit_map;
        const double threshold = mask_thresh * 255;
        cv::threshold(cbuf_map, bit_map, threshold, 255, cv::THRESH_BINARY);        

        vector<vector<vector<int>>> box_array;
        boxes_from_bitmap(pred_map, bit_map, box_array, box_thresh, unclip_ratio, min_size, max_candidates);
        float ratio_h = dst_h / (float)src_h;
        float ratio_w = dst_w / (float)src_w;

        boxes = filter_boxes(box_array, src_h, src_w, ratio_h, ratio_w);
    }
};
};