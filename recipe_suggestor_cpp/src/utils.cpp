#include "utils.hpp"
#include <sys/types.h>
#include <cstring>

Consumable get_object_from_id(u_int16_t id) {
    // TODO
    return {};
}

cv::Mat screen_capture_to_mat(const ScreenCapture& capture) {
    // Create a Mat object with the specified dimensions
    // Assuming the data is in BGR format (3 channels, 8-bit per channel)
    cv::Mat mat(capture.height, capture.width, CV_8UC3);
    
    // Copy the data from the struct to the Mat
    std::memcpy(mat.data, capture.data, capture.width * capture.height * 3);
    
    return mat;
}