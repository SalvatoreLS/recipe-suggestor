#include <filesystem>
#include <iostream>
#include "pipeline/nodes/frame_capturer.hpp"
#include "utils.hpp"

int main() {
    try {
        FrameCapturer capturer;
        std::cout << "FrameCapturer initialized." << std::endl;

        ScreenCapture* cap = capturer.capture_screen();
        if (!cap) {
            std::cerr << "Failed to capture frame." << std::endl;
            return 1;
        }

        std::cout << "Captured frame: " << cap->width << "x" << cap->height << std::endl;

        cv::Mat mat = screen_capture_to_mat(*cap);
        if (mat.empty() || mat.cols != cap->width || mat.rows != cap->height) {
            std::cerr << "Capture did not convert to a well-formed Mat." << std::endl;
            delete cap;
            return 1;
        }

        // A real desktop is never a single flat colour; this catches the
        // sheared/garbage output the old 4-bytes-in / 3-bytes-out mismatch gave.
        double min_v, max_v;
        cv::Mat gray;
        cv::cvtColor(mat, gray, cv::COLOR_BGR2GRAY);
        cv::minMaxLoc(gray, &min_v, &max_v);
        if (max_v - min_v < 1.0) {
            std::cerr << "Captured image is a flat colour -- capture is broken." << std::endl;
            delete cap;
            return 1;
        }

        std::filesystem::create_directories("outputs");
        cv::imwrite("outputs/capture.png", mat);
        std::cout << "Wrote outputs/capture.png -- open it to confirm it is your desktop." << std::endl;

        delete cap;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
