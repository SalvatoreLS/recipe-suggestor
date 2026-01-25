#ifndef FLOOR_DETECTOR_CPP
#define FLOOR_DETECTOR_CPP

#include "detector.hpp"


#include <map>

class FloorDetector : public Detector {
public:
    FloorDetector(const std::string& model_path, int wid, int hei) 
        : Detector(model_path, wid, hei) {}

    std::map<types::ConsumableID, types::Quantity> detect_floor(const cv::Mat& frame, std::vector<Prediction>* out_predictions = nullptr);
};

#endif // FLOOR_DETECTOR_CPP