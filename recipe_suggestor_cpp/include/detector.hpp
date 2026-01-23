#pragma once

#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <onnxruntime_cxx_api.h>
#include "data_structures/circular_list.hpp"
#include "constants.hpp"
#include "types.hpp"

struct Prediction {
    cv::Rect bbox;
    int classId;
    float confidence;
    // Add any other necessary fields if needed
};

class Detector {
public:
    Detector(const std::string& model_path, int wid, int hei);
    
    // Returns a circular list of formatted strings for UI/Logging
    cust::CircularList<types::ItemID> detect(const cv::Mat& frame, std::vector<Prediction>* out_predictions = nullptr);
    
    // Visualization helper
    cv::Mat visualize_detections(const cv::Mat& frame, const std::vector<Prediction>& predictions, const std::vector<std::string>& class_names);

private:
    int img_width;
    int img_height;
    
    // ONNX Runtime members
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;
    
    std::string inputName;
    std::string outputName;
    std::vector<int64_t> inputNodeDims;
    
    // Helper methods
    cv::Mat _preprocess_image(const cv::Mat& frame);
    std::vector<Prediction> _filter_predictions(const std::vector<float>& output, const std::vector<int64_t>& shape, const cv::Size& img_size, float conf_threshold = constants::default_conf_threshold);
    void _sort_bboxes(std::vector<Prediction>& predictions, int threshold = constants::sort_threshold); // Threshold for row grouping
};
