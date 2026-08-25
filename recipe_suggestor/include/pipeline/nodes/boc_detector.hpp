#ifndef BOC_DETECTOR_CPP
#define BOC_DETECTOR_CPP

#include "detector.hpp"

class BoCDetector : public Detector {
public:
    // The BoC dataset is a wide bag strip black-padded to a square, so the
    // live crop must be padded the same way rather than squashed.
    BoCDetector(const std::string& model_path, int wid, int hei)
        : Detector(model_path, wid, hei, types::Preprocess::LetterboxBlack) {}
};

#endif // BOC_DETECTOR_CPP