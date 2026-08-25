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
};

class Detector {
public:
    Detector(const std::string& model_path, int wid, int hei,
             types::Preprocess preprocess = types::Preprocess::Stretch);
    virtual ~Detector() = default;

    // Ordered left-to-right, top-to-bottom: the reading order of the bag.
    cust::CircularList<types::ConsumableID> detect(const cv::Mat& frame, std::vector<Prediction>* out_predictions = nullptr);

    // Class names as reported by the model itself. Both models are nc:21 with
    // different class lists, so their output tensors are indistinguishable by
    // shape -- this is the only way to tell which model was actually loaded.
    const std::vector<std::string>& class_names() const { return class_names_; }

    cv::Mat visualize_detections(const cv::Mat& frame, const std::vector<Prediction>& predictions, const std::vector<std::string>& class_names);
    cv::Mat visualize_detections(const cv::Mat& frame, const std::vector<Prediction>& predictions) {
        return visualize_detections(frame, predictions, class_names_);
    }

protected:
    int img_width;
    int img_height;

    // ONNX Runtime members
    Ort::Env env;
    Ort::Session session;
    Ort::AllocatorWithDefaultOptions allocator;

    std::string inputName;
    std::string outputName;
    std::vector<int64_t> inputNodeDims;
    std::vector<std::string> class_names_;

    // Letterbox geometry of the most recent preprocess, used to map boxes back.
    // Model-space -> source-space mapping produced by _preprocess_image.
    // Letterbox keeps scale_x_ == scale_y_ and may set padding; stretch uses
    // independent axis scales and no padding.
    types::Preprocess preprocess_;
    float scale_x_ = 1.0f;
    float scale_y_ = 1.0f;
    int pad_x_ = 0;
    int pad_y_ = 0;

    // Preprocess -> tensor -> session.Run -> filter. Shared by every detector
    // so the pre/post-processing contract exists in exactly one place.
    std::vector<Prediction> run_inference(const cv::Mat& frame, float conf_threshold = constants::default_conf_threshold);

    cv::Mat _preprocess_image(const cv::Mat& frame);
    std::vector<Prediction> _filter_predictions(const std::vector<float>& output, const std::vector<int64_t>& shape, const cv::Size& img_size, float conf_threshold = constants::default_conf_threshold);
    void _sort_bboxes(std::vector<Prediction>& predictions, int threshold = constants::sort_threshold); // Threshold for row grouping

private:
    void load_class_names();
};
