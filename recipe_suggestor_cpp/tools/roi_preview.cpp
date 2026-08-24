// ROI calibration helper. Takes a full-resolution screenshot, runs it through the
// real Router, and writes an overlay showing where the bag crop and the masked
// HUD strip land, plus the two images the detectors would actually receive.
//
//   ./roi_preview <screenshot> [out_dir]
//
// Not a test: it is a human-in-the-loop tool for tuning the constants::*_factor
// values against a real screen. Run it, look at outputs/roi_overlay.png, adjust.
#include <filesystem>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "constants.hpp"
#include "pipeline/nodes/boc_detector.hpp"
#include "pipeline/nodes/floor_detector.hpp"
#include "pipeline/nodes/router.hpp"
#include "utils.hpp"

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <screenshot> [out_dir]\n";
        return 2;
    }
    const std::string out_dir = (argc > 2) ? argv[2] : "outputs";

    cv::Mat frame = cv::imread(argv[1], cv::IMREAD_COLOR);
    if (frame.empty()) {
        std::cerr << "Could not read " << argv[1] << "\n";
        return 1;
    }
    std::cout << "Frame: " << frame.cols << "x" << frame.rows << "\n";

    ScreenCapture source;
    source.width = frame.cols;
    source.height = frame.rows;
    const size_t n = static_cast<size_t>(frame.cols) * frame.rows * 3;
    source.data = new unsigned char[n];
    if (frame.isContinuous()) {
        std::memcpy(source.data, frame.data, n);
    } else {
        for (int y = 0; y < frame.rows; ++y)
            std::memcpy(source.data + static_cast<size_t>(y) * frame.cols * 3,
                        frame.ptr(y), static_cast<size_t>(frame.cols) * 3);
    }

    Router router;
    ScreenCapture floor_img = {nullptr, 0, 0};
    ScreenCapture boc_img = {nullptr, 0, 0};
    router.route(&source, &floor_img, &boc_img);

    const cv::Rect boc = router.boc_rect(frame.cols, frame.rows);
    const cv::Rect hud = router.left_rect(frame.cols, frame.rows);
    // The bag strip is black-padded to the model square, so what matters is
    // whether its aspect matches the training band's 640:170 = 3.765.
    std::cout << "boc_rect : " << boc << "  aspect "
              << (double)boc.width / boc.height << " (training band is 3.765)\n"
              << "left_rect: " << hud << "\n";

    cv::Mat overlay = frame.clone();
    cv::rectangle(overlay, boc, cv::Scalar(0, 255, 0), 3);
    cv::rectangle(overlay, hud, cv::Scalar(0, 0, 255), 3);
    cv::putText(overlay, "bag crop", {boc.x, std::max(20, boc.y - 8)},
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 255, 0), 2);
    cv::putText(overlay, "HUD mask (floor blind spot)", {hud.x + 4, hud.y - 8},
                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);

    // Run the real detectors when the models are on disk, so the overlay shows
    // not just where the regions are but whether anything is actually found in
    // them. This is the whole point of calibrating: a rect that looks right and
    // detects nothing is still wrong.
    std::filesystem::create_directories(out_dir);
    cv::Mat boc_mat = screen_capture_to_mat(boc_img);
    cv::Mat floor_mat = screen_capture_to_mat(floor_img);

    if (std::filesystem::exists(constants::boc_model_path)) {
        try {
            BoCDetector boc_det(constants::boc_model_path, constants::img_width, constants::img_height);
            std::vector<Prediction> preds;
            boc_det.detect(boc_mat, &preds);
            std::cout << "bag detections (" << preds.size() << "):";
            for (const auto& p : preds)
                std::cout << " " << boc_det.class_names()[p.classId]
                          << cv::format("(%.2f)", p.confidence);
            std::cout << "\n";
            cv::imwrite(out_dir + "/roi_boc_detected.png", boc_det.visualize_detections(boc_mat, preds));
        } catch (const std::exception& e) {
            std::cerr << "BoC detector unavailable: " << e.what() << "\n";
        }
    }
    if (std::filesystem::exists(constants::floor_model_path)) {
        try {
            FloorDetector floor_det(constants::floor_model_path, constants::img_width, constants::img_height);
            std::vector<Prediction> preds;
            floor_det.detect_floor(floor_mat, &preds);
            std::cout << "floor detections (" << preds.size() << "):";
            for (const auto& p : preds)
                std::cout << " " << floor_det.class_names()[p.classId]
                          << cv::format("(%.2f)", p.confidence);
            std::cout << "\n";
            cv::imwrite(out_dir + "/roi_floor_detected.png", floor_det.visualize_detections(floor_mat, preds));
        } catch (const std::exception& e) {
            std::cerr << "Floor detector unavailable: " << e.what() << "\n";
        }
    }
    cv::imwrite(out_dir + "/roi_overlay.png", overlay);
    cv::imwrite(out_dir + "/roi_boc.png", boc_mat);
    cv::imwrite(out_dir + "/roi_floor.png", floor_mat);
    std::cout << "Wrote " << out_dir << "/roi_overlay.png, roi_boc.png, roi_floor.png\n";
    return 0;
}
