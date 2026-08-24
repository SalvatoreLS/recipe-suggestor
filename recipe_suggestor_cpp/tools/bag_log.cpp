// Read the crafting bag out of captured frames, as component ids.
//
//   ./bag_log frame1.png frame2.png ...
//
// Prints one line per frame: the file, the bag size, and the component ids the
// detector sees, already translated through class_map.json into the ids the
// crafting algorithm uses. This exists so an observation for find_start_seed is
// read off the screen rather than typed from memory -- a single mistyped
// component invalidates the whole constraint and quietly eliminates the true
// seed.
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

#include "constants.hpp"
#include "pipeline/nodes/boc_detector.hpp"
#include "pipeline/nodes/recipe_suggestor.hpp"
#include "pipeline/nodes/router.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <frame.png> [more frames...]\n";
        return 2;
    }

    BoCDetector detector(constants::boc_model_path, constants::img_width, constants::img_height);
    RecipeSuggestor suggestor;
    suggestor.bind_models(detector.class_names(), {});
    Router router;

    for (int i = 1; i < argc; ++i) {
        cv::Mat frame = cv::imread(argv[i]);
        if (frame.empty()) {
            std::cerr << argv[i] << ": cannot read\n";
            continue;
        }

        cv::Mat bag_crop = frame;
        ScreenCapture source;
        ScreenCapture floor_img = {nullptr, 0, 0};
        ScreenCapture boc_img = {nullptr, 0, 0};
        if (frame.cols != constants::img_width || frame.rows != constants::img_height) {
            source.width = frame.cols;
            source.height = frame.rows;
            const size_t n = static_cast<size_t>(frame.cols) * frame.rows * 3;
            source.data = new unsigned char[n];
            std::memcpy(source.data, frame.data, n);
            router.route(&source, &floor_img, &boc_img);
            bag_crop = screen_capture_to_mat(boc_img);
        }

        std::vector<Prediction> preds;
        auto detected = detector.detect(bag_crop, &preds);
        const auto bag = suggestor.translate_bag(detected.snapshot());

        std::cout << argv[i] << "  " << bag.size() << "/8  ";
        for (size_t k = 0; k < bag.size(); ++k) std::cout << (k ? " " : "") << bag[k];
        std::cout << "   # ";
        for (size_t k = 0; k < bag.size(); ++k)
            std::cout << (k ? ", " : "") << suggestor.consumable_name(bag[k]);
        std::cout << "\n";
    }
    return 0;
}
