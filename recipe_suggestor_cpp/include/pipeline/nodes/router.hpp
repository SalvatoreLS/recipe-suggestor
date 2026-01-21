#ifndef ROUTER_HPP
#define ROUTER_HPP

#include <utility>
#include <thread>
#include "utils.hpp"

class Router {

public:
    Router();
    void route(ScreenCapture* source_img, ScreenCapture* floor_img, ScreenCapture* boc_img); // update the provided object

private:
    void process_boc(ScreenCapture* source, ScreenCapture* dest);
    void process_floor(ScreenCapture* source, ScreenCapture* dest);
};

#endif // ROUTER_HPP