#ifndef FLOOR_DETECTOR_CPP
#define FLOOR_DETECTOR_CPP

#include "detector.hpp"

class FloorDetector : public Detector {
public:
    FloorDetector(const std::string& model_path, int wid, int hei) 
        : Detector(model_path, wid, hei) {}
};

#endif // FLOOR_DETECTOR_CPP