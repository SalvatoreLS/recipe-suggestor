#include "pipeline/nodes/floor_detector.hpp"
#include "constants.hpp"

// The floor pass only cares about "what is lying around, and how much of it",
// so it counts classes instead of preserving reading order. Everything up to
// that point is Detector::run_inference, shared with the bag pass.
std::map<types::ConsumableID, types::Quantity> FloorDetector::detect_floor(const cv::Mat& frame, std::vector<Prediction>* out_predictions) {
    std::vector<Prediction> predictions = run_inference(frame, constants::floor_conf_threshold);

    if (out_predictions) *out_predictions = predictions;

    std::map<types::ConsumableID, types::Quantity> counts;
    for (const auto& pred : predictions) {
        auto id = static_cast<types::ConsumableID>(pred.classId);
        if (counts[id] < 255) counts[id]++;
    }
    return counts;
}
