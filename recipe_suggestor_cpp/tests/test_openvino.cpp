#include <opencv2/dnn.hpp>
#include <iostream>

/*
g++ -std=c++17 test_openvino.cpp -o test_openvino \
    `pkg-config --cflags --libs opencv4`

./test_openvino
*/

int main() {
    cv::dnn::Net net;
    
    // Check if OpenVINO backend is available
    std::vector<std::pair<cv::dnn::Backend, cv::dnn::Target>> backends = cv::dnn::getAvailableBackends();
    
    std::cout << "Available DNN backends:" << std::endl;
    for (const auto& backend : backends) {
        std::cout << "  Backend: " << backend.first 
                  << ", Target: " << backend.second << std::endl;
    }
    
    // Check specifically for Inference Engine
    bool hasInferenceEngine = false;
    for (const auto& backend : backends) {
        if (backend.first == cv::dnn::DNN_BACKEND_INFERENCE_ENGINE) {
            hasInferenceEngine = true;
            std::cout << "\n✓ OpenVINO Inference Engine is AVAILABLE!" << std::endl;
            break;
        }
    }
    
    if (!hasInferenceEngine) {
        std::cout << "\n✗ OpenVINO Inference Engine is NOT available" << std::endl;
        return 1;
    }
    
    return 0;
}