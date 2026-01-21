#include <iostream>
#include <opencv2/opencv.hpp>
#include <filesystem>
#include "pipeline/nodes/boc_detector.hpp"

namespace fs = std::filesystem;

int main() {
    std::string model_path = "resources/models/best.onnx";
    std::string image_path = "resources/test_images/test1.jpg";
    std::string output_path = "outputs/boc_test_result.jpg";

    if (!fs::exists(model_path)) {
        std::cerr << "Model not found: " << model_path << std::endl;
        return 1;
    }
    if (!fs::exists(image_path)) {
        std::cerr << "Image not found: " << image_path << std::endl;
        return 1;
    }

    // Initialize BoC Detector
    // Assuming model input size is 640x640 based on typical usage
    BoCDetector detector(model_path, 640, 640);

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

    // Visualize
    std::vector<std::string> class_names; // Mock class names or load from file if available
    for(int i=0; i<80; ++i) class_names.push_back(std::to_string(i));

    cv::Mat out_frame = detector.visualize_detections(frame, predictions, class_names);

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
