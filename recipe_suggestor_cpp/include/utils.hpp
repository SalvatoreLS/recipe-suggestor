#ifndef UTILS_HPP
#define UTILS_HPP

#include <sys/types.h>
#include <string>
#include <opencv2/opencv.hpp>


struct ScreenCapture {
    unsigned char* data;
    u_int16_t width;
    u_int16_t height;

    ~ScreenCapture() { delete[] data; }
};

struct Consumable {
    u_int16_t id;
    std::string name;
};

Consumable get_object_from_id(u_int16_t id);
cv::Mat screen_capture_to_mat(const ScreenCapture& capture);

#endif // UTILS_HPP