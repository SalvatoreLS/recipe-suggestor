#include <iostream>
#include <opencv2/opencv.hpp>
#include "pipeline/nodes/floor_detector.hpp"

int main() {
    std::string model_path = "resources/models/best.onnx"; 
    
    // Mock image (black image)
    cv::Mat frame = cv::Mat::zeros(640, 640, CV_8UC3);
    
    FloorDetector detector(model_path, 640, 640);
    
    std::cout << "Running Floor detection on blank image..." << std::endl;
    
    std::vector<Prediction> predictions;
    detector.detect(frame, &predictions);
    
    std::cout << "Floor detection run complete. Found: " << predictions.size() << " objects (expected 0)." << std::endl;
    
    return 0;
}
