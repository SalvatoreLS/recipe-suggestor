#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include "data_structures/circular_list.hpp"

#ifdef USE_LIBTORCH
#include <torch/torch.h>
#include <torch/script.h>
#endif

struct Prediction {
    cv::Rect bbox;
    int classId;
    float confidence;
};

class Detector {
public:
    Detector(const std::string& model_path_prefix, int wid, int hei);
    
    cust::CircularList<std::string> detect(const cv::Mat& frame, std::vector<Prediction>* out_predictions = nullptr);
    cv::Mat visualize_detections(const cv::Mat& frame, const std::vector<Prediction>& predictions, const std::vector<std::string>& class_names);

private:
    int img_width;
    int img_height;

    cv::Mat _preprocess_image(const cv::Mat& frame);
    void _sort_bboxes(std::vector<Prediction>& predictions, int threshold = 10);

#ifdef USE_LIBTORCH
    torch::jit::script::Module model;
    std::vector<std::string> class_names;
    torch::Tensor _mat_to_tensor(const cv::Mat& mat);
    std::vector<Prediction> _filter_predictions(const torch::Tensor& output, const cv::Size& img_size, float conf_threshold = 0.5f);
#else
    cv::dnn::Net net;
    std::vector<Prediction> _filter_predictions(const cv::Mat& output, const cv::Size& img_size, float conf_threshold = 0.5f);
#endif
};
