#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "pipeline/nodes/boc_detector.hpp"
#include "constants.hpp"

namespace fs = std::filesystem;

int main() {
    std::string model_path = constants::boc_model_path;
    std::string image_path = "resources/test_images/C00144_png.rf.a60cb305b62d17bb303017d841b5c12b.jpg";
    std::string output_path = "outputs/boc_test_result.jpg";

    // Both inputs are out-of-tree: the model is build output and the image is a
    // dataset crop, neither of which a fresh clone has. Skip rather than fail, the
    // way floor_test and screen_regression_test already do.
    if (!fs::exists(model_path)) {
        std::cout << "boc_test skipped: " << model_path << " not present" << std::endl;
        return 0;
    }
    if (!fs::exists(image_path)) {
        std::cout << "boc_test skipped: " << image_path << " not present" << std::endl;
        return 0;
    }

    // Initialize BoC Detector
    // Assuming model input size is 640x640 based on typical usage
    BoCDetector detector(model_path, constants::img_width, constants::img_height);

    // Load Image
    cv::Mat frame = cv::imread(image_path);
    if (frame.empty()) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }

    std::cout << "Running BoC detection on " << image_path << "..." << std::endl;

    // Detect
    std::vector<Prediction> predictions;
    auto results = detector.detect(frame, &predictions);

    std::cout << "Detected " << predictions.size() << " objects." << std::endl;

    // Class names come from the model's own metadata, so the labels drawn here
    // are guaranteed to belong to the model that produced the boxes.
    const auto& class_names = detector.class_names();
    if (class_names.empty()) {
        std::cerr << "Model reported no class names in its metadata." << std::endl;
        return 1;
    }
    std::cout << "Model reports " << class_names.size() << " classes, first: "
              << class_names.front() << ", last: " << class_names.back() << std::endl;

    for (const auto& pred : predictions) {
        std::cout << "  " << (pred.classId < (int)class_names.size() ? class_names[pred.classId] : "?")
                  << " " << cv::format("%.2f", pred.confidence)
                  << " at " << pred.bbox << std::endl;
    }

    cv::Mat out_frame = detector.visualize_detections(frame, predictions);

    // Save
    fs::create_directories("outputs");
    if (cv::imwrite(output_path, out_frame)) {
        std::cout << "Saved result to " << output_path << std::endl;
    } else {
        std::cerr << "Failed to save result" << std::endl;
        return 1;
    }

    return 0;
}
