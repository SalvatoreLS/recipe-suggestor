#ifndef ROUTER_HPP
#define ROUTER_HPP

#include <utility>
#include <thread>
#include "utils.hpp"
#include <cstring>
#include <algorithm>

class Router {

public:
    Router();
    void route(ScreenCapture* source_img, ScreenCapture* floor_img, ScreenCapture* boc_img); // update the provided object

    // Region geometry, shared by the crop and the mask so they cannot diverge.
    // Public so the ROI calibration tool can draw exactly what the pipeline uses.
    cv::Rect boc_rect(int w, int h) const;
    cv::Rect left_rect(int w, int h) const;

private:

    void process_boc(ScreenCapture* source, ScreenCapture* dest);
    void process_floor(ScreenCapture* source, ScreenCapture* dest);
    void cover_region_inplace(ScreenCapture* img, const cv::Rect& region);
};

#endif // ROUTER_HPP
