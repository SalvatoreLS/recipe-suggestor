#ifndef FLOOR_DETECTOR_CPP
#define FLOOR_DETECTOR_CPP

#include "detector.hpp"


#include <map>

class FloorDetector : public Detector {
public:
    // The floor dataset is whole frames squashed to 640x640 by Roboflow.
    FloorDetector(const std::string& model_path, int wid, int hei)
        : Detector(model_path, wid, hei, types::Preprocess::Stretch) {}

    std::map<types::ConsumableID, types::Quantity> detect_floor(const cv::Mat& frame, std::vector<Prediction>* out_predictions = nullptr);
};

#endif // FLOOR_DETECTOR_CPP