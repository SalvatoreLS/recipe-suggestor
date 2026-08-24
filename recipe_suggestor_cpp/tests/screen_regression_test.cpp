// End-to-end gate on real screen frames.
//
// Everything else in the suite runs on dataset crops, which are exactly the
// distribution the models were fitted to. These two frames are raw 1920x1080
// captures of the running game, so they exercise the part that dataset crops
// cannot: the Router's strip geometry, the per-model preprocessing, and the
// masks -- against a ground truth read off the screen by eye.
//
// If a change to constants::*_factor or to types::Preprocess breaks the live
// path, this is what fails.
#include <filesystem>
#include <iostream>
#include <map>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "constants.hpp"
#include "pipeline/nodes/boc_detector.hpp"
#include "pipeline/nodes/floor_detector.hpp"
#include "pipeline/nodes/router.hpp"
#include "utils.hpp"

namespace fs = std::filesystem;

static int failures = 0;
static void check(bool cond, const std::string& what) {
    std::cout << (cond ? "  ok   " : "  FAIL ") << what << std::endl;
    if (!cond) failures++;
}

using Counts = std::map<std::string, int>;

static Counts named_counts(const std::vector<Prediction>& preds, const std::vector<std::string>& names) {
    Counts c;
    for (const auto& p : preds)
        c[p.classId < (int)names.size() ? names[p.classId] : "?"]++;
    return c;
}

static std::string describe(const Counts& c) {
    std::string s;
    for (const auto& [n, k] : c) {
        if (!s.empty()) s += ", ";
        s += n + " x" + std::to_string(k);
    }
    return s.empty() ? "(nothing)" : s;
}

struct Frame {
    std::string path;
    Counts expected_bag;
    Counts expected_floor;
};

int main() {
    if (!fs::exists(constants::boc_model_path) || !fs::exists(constants::floor_model_path)) {
        std::cout << "screen_regression_test skipped: a model is missing" << std::endl;
        return 0;
    }

    // Ground truth read off the screenshots by eye.
    const std::vector<Frame> frames = {
        // Bag holds a penny and a key; the floor has two coins and a bomb. The
        // chest and the poop are not consumables and must not be reported.
        {"resources/test_images/screens/gameplay_bag_penny_key.png",
         {{"penny", 1}, {"key", 1}},
         {{"penny", 2}, {"bomb", 1}}},
        // Paused, empty bag, room hidden behind the menu: the honest answer is
        // nothing at all. Guards against the menu art tripping either model.
        {"resources/test_images/screens/paused_empty_bag.png", {}, {}},
    };

    BoCDetector boc_det(constants::boc_model_path, constants::img_width, constants::img_height);
    FloorDetector floor_det(constants::floor_model_path, constants::img_width, constants::img_height);
    Router router;

    for (const auto& f : frames) {
        std::cout << f.path << std::endl;
        cv::Mat frame = cv::imread(f.path);
        if (frame.empty()) {
            check(false, "frame loads");
            continue;
        }
        check(frame.cols == 1920 && frame.rows == 1080, "frame is a full 1920x1080 capture");

        ScreenCapture source;
        source.width = frame.cols;
        source.height = frame.rows;
        const size_t n = static_cast<size_t>(frame.cols) * frame.rows * 3;
        source.data = new unsigned char[n];
        std::memcpy(source.data, frame.data, n);

        ScreenCapture floor_img = {nullptr, 0, 0};
        ScreenCapture boc_img = {nullptr, 0, 0};
        router.route(&source, &floor_img, &boc_img);

        std::vector<Prediction> bag_preds, floor_preds;
        boc_det.detect(screen_capture_to_mat(boc_img), &bag_preds);
        floor_det.detect_floor(screen_capture_to_mat(floor_img), &floor_preds);

        const Counts bag = named_counts(bag_preds, boc_det.class_names());
        const Counts floor = named_counts(floor_preds, floor_det.class_names());
        std::cout << "    bag   : " << describe(bag) << "  (expected " << describe(f.expected_bag) << ")\n"
                  << "    floor : " << describe(floor) << "  (expected " << describe(f.expected_floor) << ")\n";

        check(bag == f.expected_bag, "bag contents match the ground truth");
        check(floor == f.expected_floor, "floor pickups match the ground truth");
    }

    // --- the bag is a FIFO queue -------------------------------------------
    // Everything the planner says about a full bag ("pick this up, it pushes
    // THAT out") rests on this. Verified in game and pinned here: two captures
    // of the same run, one pickup apart, with the bag full in the first.
    {
        std::cout << "FIFO: full bag + one pickup" << std::endl;
        auto bag_sequence = [&](const std::string& path) {
            cv::Mat frame = cv::imread(path);
            ScreenCapture src;
            src.width = frame.cols;
            src.height = frame.rows;
            const size_t n = static_cast<size_t>(frame.cols) * frame.rows * 3;
            src.data = new unsigned char[n];
            std::memcpy(src.data, frame.data, n);
            ScreenCapture floor_img = {nullptr, 0, 0}, boc_img = {nullptr, 0, 0};
            router.route(&src, &floor_img, &boc_img);
            std::vector<Prediction> preds;
            boc_det.detect(screen_capture_to_mat(boc_img), &preds);
            std::vector<std::string> names;
            for (const auto& p : preds) names.push_back(boc_det.class_names()[p.classId]);
            return names;
        };

        // Detections come back in the bag's reading order, left to right and
        // top to bottom, which is the order the slots fill in.
        const auto before = bag_sequence("resources/test_images/screens/fifo_before_full_bag.png");
        const auto after = bag_sequence("resources/test_images/screens/fifo_after_heart_pickup.png");

        auto join = [](const std::vector<std::string>& v) {
            std::string s;
            for (size_t i = 0; i < v.size(); ++i) { if (i) s += " "; s += v[i]; }
            return s;
        };
        std::cout << "    before: " << join(before) << "\n    after : " << join(after) << std::endl;

        check(before.size() == constants::bag_size, "the bag was full before the pickup");
        check(after.size() == constants::bag_size, "the bag is still full after it");

        if (before.size() == constants::bag_size && after.size() == constants::bag_size) {
            // FIFO: the oldest entry falls off the front, the new one lands at
            // the back. Any other model (newest-out, or pickups blocked while
            // full) would fail one of these.
            std::vector<std::string> predicted(before.begin() + 1, before.end());
            predicted.push_back("red_heart");
            check(after == predicted, "picking up past a full bag drops the OLDEST entry and appends the new one");
            check(after.front() == before[1], "the front advanced by exactly one slot");
            check(after.back() == "red_heart", "the picked-up Red Heart is at the back");
        }
    }

    std::cout << (failures ? "screen_regression_test FAILED" : "screen_regression_test passed") << std::endl;
    return failures ? 1 : 0;
}
