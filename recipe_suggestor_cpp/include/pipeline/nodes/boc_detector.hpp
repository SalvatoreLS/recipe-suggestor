#ifndef BOC_DETECTOR_CPP
#define BOC_DETECTOR_CPP

#include "detector.hpp"

class BoCDetector : public Detector {
public:
    BoCDetector(const std::string& model_path, int wid, int hei) 
        : Detector(model_path, wid, hei) {}
};

#endif // BOC_DETECTOR_CPP