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

private:
    void process_boc(ScreenCapture* source, ScreenCapture* dest);
    void process_floor(ScreenCapture* source, ScreenCapture* dest);
    void cover_region_inplace(ScreenCapture* img, uint16_t x1, uint16_t y1, uint16_t x2, uint16_t y2);
};

#endif // ROUTER_HPP