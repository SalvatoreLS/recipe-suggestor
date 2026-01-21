#include <iostream>
#include "pipeline/nodes/frame_capturer.hpp"
#include "utils.hpp"

int main() {
    try {
        FrameCapturer capturer;
        std::cout << "FrameCapturer initialized." << std::endl;

        ScreenCapture* cap = capturer.capture_screen();
        if (cap) {
            std::cout << "Captured frame: " << cap->width << "x" << cap->height << std::endl;
            delete cap;
            return 0;
        } else {
            std::cerr << "Failed to capture frame." << std::endl;
            return 1;
        }
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
