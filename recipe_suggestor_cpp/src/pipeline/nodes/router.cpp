#include "pipeline/nodes/router.hpp"
#include "constants.hpp"
#include <sys/types.h>

Router::Router() {}

void Router::route(ScreenCapture* source_img, ScreenCapture* floor_img, ScreenCapture* boc_img) {
    std::thread boc_thread(&Router::process_boc, this, source_img, boc_img);
    std::thread floor_thread(&Router::process_floor, this, source_img, floor_img);

    boc_thread.join(); floor_thread.join();
}

// Single source of truth for where the crafting bag sits on screen. Both the
// BoC crop and the mask the floor pass paints over it derive from this, so the
// two can no longer disagree.
cv::Rect Router::boc_rect(int w, int h) const {
    int crop_w = static_cast<int>(w * constants::boc_crop_width_factor);
    int crop_h = static_cast<int>(h * constants::boc_crop_height_factor);
    int x = static_cast<int>(w * constants::crop_start_x_factor) - crop_w;
    int y = h - crop_h;
    return cv::Rect(x, y, crop_w, crop_h) & cv::Rect(0, 0, w, h);
}

// The left-hand HUD strip (health, coins, keys), which must not be mistaken
// for pickups lying on the floor.
cv::Rect Router::left_rect(int w, int h) const {
    int x = static_cast<int>(w * constants::left_section_x_factor);
    int y = static_cast<int>(h * constants::left_section_y_factor);
    int rw = static_cast<int>(w * constants::left_crop_width_factor);
    int rh = static_cast<int>(h * constants::left_crop_height_factor);
    return cv::Rect(x, y, rw, rh) & cv::Rect(0, 0, w, h);
}

void Router::process_boc(ScreenCapture* source, ScreenCapture* dest) {
    cv::Rect roi = boc_rect(source->width, source->height);

    dest->width = static_cast<u_int16_t>(roi.width);
    dest->height = static_cast<u_int16_t>(roi.height);
    dest->data = new unsigned char[static_cast<size_t>(roi.width) * roi.height * 3];
    if (roi.width == 0 || roi.height == 0) return;

    // Packed BGR24 in, packed BGR24 out (see the ScreenCapture invariant).
    cv::Mat src(source->height, source->width, CV_8UC3, source->data);
    cv::Mat out(roi.height, roi.width, CV_8UC3, dest->data);
    src(roi).copyTo(out);
}

void Router::process_floor(ScreenCapture* source, ScreenCapture* dest) {
    // 1. Copy the full frame
    dest->width = source->width;
    dest->height = source->height;

    size_t dataSize = static_cast<size_t>(source->width) * source->height * 3;
    dest->data = new unsigned char[dataSize];
    std::memcpy(dest->data, source->data, dataSize);

    // 2. Mask out the crafting bag and the left HUD strip
    cover_region_inplace(dest, boc_rect(source->width, source->height));
    cover_region_inplace(dest, left_rect(source->width, source->height));
}

void Router::cover_region_inplace(ScreenCapture* img, const cv::Rect& region) {
    cv::Rect r = region & cv::Rect(0, 0, img->width, img->height);
    if (r.width <= 0 || r.height <= 0) return;

    cv::Mat mat(img->height, img->width, CV_8UC3, img->data);
    mat(r).setTo(cv::Scalar(0, 0, 0));
}
