#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "pipeline/nodes/floor_detector.hpp"
#include "constants.hpp"
#include "utils.hpp"

int main() {
    const std::string model_path = constants::floor_model_path;

    // The floor model is trained separately and may legitimately be absent.
    // Skipping keeps the suite green instead of aborting the whole run.
    if (!std::filesystem::exists(model_path)) {
        std::cout << "**********************************************************\n"
                  << "SKIP: " << model_path << " is not present, so floor\n"
                  << "detection is NOT covered by this run. Train it with:\n"
                  << "  python3 models_training/train_floor.py --data pickups---TBOI-2/data.yaml\n"
                  << "**********************************************************" << std::endl;
        return 0;
    }

    FloorDetector detector(model_path, constants::img_width, constants::img_height);

    // The wrong-model tripwire. Both detectors emit an identically shaped
    // tensor, so a swapped file mislabels everything silently; the only real
    // guard is the class list the model reports in its own ONNX metadata.
    const auto& names = detector.class_names();
    std::cout << "Model reports " << names.size() << " classes." << std::endl;
    try {
        load_class_map(constants::class_map_path, "floor", names);
    } catch (const std::exception& e) {
        std::cerr << "Floor model does not match class_map.json: " << e.what() << std::endl;
        return 1;
    }

    cv::Mat frame = cv::Mat::zeros(constants::img_height, constants::img_width, CV_8UC3);

    std::cout << "Running Floor detection on blank image..." << std::endl;

    std::vector<Prediction> predictions;
    auto floor_items = detector.detect_floor(frame, &predictions);

    std::cout << "Floor detection run complete. Found: " << predictions.size()
              << " objects (expected 0), " << floor_items.size() << " distinct classes." << std::endl;

    if (!predictions.empty()) {
        std::cerr << "Expected no detections on a blank image." << std::endl;
        return 1;
    }
    return 0;
}
