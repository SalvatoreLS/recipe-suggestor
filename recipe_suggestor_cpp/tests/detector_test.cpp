#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core.hpp>

#include "detector.hpp"

namespace {
namespace fs = std::filesystem;

std::vector<std::string> load_class_names(const fs::path& txt_path) {
    std::vector<std::string> names;
    std::ifstream in(txt_path);
    if (!in.is_open()) {
        return names;
    }
    std::string line;
    while (std::getline(in, line)) {
        if (!line.empty()) {
            names.push_back(line);
        }
    }
    return names;
}
}

int main(int argc, char** argv) {

    // Input and output
    const std::string images_path = "../resources/test_images/";
    const std::string save_path = "../outputs/";

    // ONNX model
    const std::string model_path = "../resources/models/best.onnx";
    const std::string labels_path = "../resources/models/synset.txt";

    const int width = 640;
    const int height = 640;

    std::cout << "Detector runtime test starting...\n";
    std::cout << "Model hint: " << model_path << "\n";
    std::cout << "Images path: " << images_path << "\n";

    // Check if path exists
    fs::path images_dir = fs::path(images_path);
    if (!fs::exists(images_dir)) {
        std::cerr << "Error: test image path does not exist: '" << images_dir << "'\n";
        return 1;
    }
    
    if (!fs::is_directory(images_dir)) {
        std::cerr << "Error: test image path is not a directory: '" << images_dir << "'\n";
        return 1;
    }
    
    // Validate at least one image exists
    bool has_valid_image = false;
    for (const auto& entry : fs::directory_iterator(images_dir)) {
        if (entry.is_regular_file()) {
            cv::Mat test_image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
            if (!test_image.empty()) {
                has_valid_image = true;
                break;
            }
        }
    }
    
    if (!has_valid_image) {
        std::cerr << "Error: No valid images found in directory\n";
        return 1;
    }

    // Loop over all images in the directory and run detection
    for (const auto& entry : fs::directory_iterator(images_dir)) {
        if (!entry.is_regular_file()) continue;
        
        cv::Mat test_image = cv::imread(entry.path().string(), cv::IMREAD_COLOR);
        if (test_image.empty()) {
            std::cerr << "Warning: Skipping unreadable image: '" << entry.path().string() << "'\n";
            continue;
        }
        
        std::cout << "Processing image: " << entry.path().string() << std::endl;
        std::cout << "Image Shape: (" << test_image.rows << ", " << test_image.cols << ", " << test_image.channels() << ")" << std::endl;

        try {
            Detector detector(model_path, width, height);
            std::vector<Prediction> predictions;
            auto detected = detector.detect(test_image, &predictions);

            std::vector<std::string> class_names = load_class_names(labels_path);
            cv::Mat vis_image = detector.visualize_detections(test_image, predictions, class_names);

            fs::path save_dir = fs::path(save_path);
            if (!save_dir.empty()) {
                std::error_code ec;
                fs::create_directories(save_dir, ec);
            }

            std::string output_file = save_path + entry.path().filename().string();
            if (!cv::imwrite(output_file, vis_image)) {
                std::cerr << "Warning: could not write visualization to '" << output_file << "'\n";
            } else {
                std::cout << "Saved visualization to: " << output_file << std::endl;
            }
            
            if (predictions.empty()) {
                std::cout << "[WARNING] No objects detected in test image " << entry.path().string() << ".\n";
            }
            else {
                std::cout << "[SUCCESS] Detected " << predictions.size() << " objects in image " << entry.path().string() << ".\n";
            }

            std::cout << "[SUCCESS] Detector ran successfully. Returned label container (type: "
                << typeid(detected).name() << ")" << std::endl;

            /* TODO: ADD CLASS NAMES
            std::cout << "Detected Labels (ItemIDs): ";
            for (const auto& label_node : detected) {
                std::cout << label_node << " ";
            }
            */
           
            std::cout << std::endl;

        } catch (const cv::Exception &e) {
            std::cerr << "OpenCV exception while creating/running Detector: " << e.what() << std::endl;
            return 2;
        } catch (const Ort::Exception &e) {
            std::cerr << "ONNX Runtime Exception: " << e.what() << std::endl; 
            std::cerr << "Check that '" << model_path << "' exists and is a valid ONNX model." << std::endl;
            return 2;
        } catch (const std::exception &e) {
            std::cerr << "std::exception while creating/running Detector: " << e.what() << std::endl;
            return 3;
        } catch (...) {
            std::cerr << "Unknown exception while running Detector." << std::endl;
            return 4;
        }
    }

    std::cout << "[SUCCESS] Detector runtime test finished successfully.\n";
    return 0;
}