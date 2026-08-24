#include <cstdlib>
#include <filesystem>
#include <iostream>
#include "pipeline/nodes/router.hpp"
#include "constants.hpp"
#include "utils.hpp"

static int failures = 0;

static void check(bool cond, const std::string& what) {
    std::cout << (cond ? "  ok   " : "  FAIL ") << what << std::endl;
    if (!cond) failures++;
}

static bool is_black(const ScreenCapture& img, int x, int y) {
    size_t o = (static_cast<size_t>(y) * img.width + x) * 3;
    return img.data[o] == 0 && img.data[o + 1] == 0 && img.data[o + 2] == 0;
}

int main() {
    Router router;

    const int w = 1920;
    const int h = 1080;
    ScreenCapture source;
    source.width = w;
    source.height = h;
    source.data = new unsigned char[static_cast<size_t>(w) * h * 3];
    // Non-zero everywhere, so "black" can only come from the mask.
    for (size_t i = 0; i < static_cast<size_t>(w) * h * 3; ++i) source.data[i] = (unsigned char)(i % 254 + 1);

    ScreenCapture floor_img = {nullptr, 0, 0};
    ScreenCapture boc_img = {nullptr, 0, 0};

    std::cout << "Routing..." << std::endl;
    router.route(&source, &floor_img, &boc_img);

    std::cout << "BoC Image: " << boc_img.width << "x" << boc_img.height << std::endl;
    std::cout << "Floor Image: " << floor_img.width << "x" << floor_img.height << std::endl;

    const int exp_bw = static_cast<int>(w * constants::boc_crop_width_factor);
    const int exp_bh = static_cast<int>(h * constants::boc_crop_height_factor);
    const int exp_bx = static_cast<int>(w * constants::crop_start_x_factor) - exp_bw;
    const int exp_by = h - exp_bh;

    check(boc_img.width == exp_bw && boc_img.height == exp_bh,
          "BoC crop is the expected size (" + std::to_string(exp_bw) + "x" + std::to_string(exp_bh) + ")");
    check(floor_img.width == w && floor_img.height == h, "floor image keeps the source size");

    // The BoC region must actually be masked in the floor image. This used to
    // silently collapse to a single pixel.
    check(is_black(floor_img, exp_bx + exp_bw / 2, exp_by + exp_bh / 2), "BoC region is masked in the floor image");
    check(is_black(floor_img, exp_bx + 1, exp_by + 1), "BoC mask covers its top-left corner");
    check(is_black(floor_img, exp_bx + exp_bw - 2, h - 2), "BoC mask covers its bottom-right corner");

    // Left HUD strip
    const int lx = static_cast<int>(w * constants::left_section_x_factor);
    const int ly = static_cast<int>(h * constants::left_section_y_factor);
    const int lw = static_cast<int>(w * constants::left_crop_width_factor);
    const int lh = static_cast<int>(h * constants::left_crop_height_factor);
    check(is_black(floor_img, lx + lw / 2, ly + lh / 2), "left HUD strip is masked");

    // A pixel outside both masks must survive untouched.
    check(!is_black(floor_img, w / 2, h / 2), "centre of the frame is left alone");

    // The crop must match the source pixels at the same offset.
    bool crop_matches = true;
    for (int y = 0; y < exp_bh && crop_matches; y += 17) {
        for (int x = 0; x < exp_bw && crop_matches; x += 17) {
            size_t s = (static_cast<size_t>(exp_by + y) * w + (exp_bx + x)) * 3;
            size_t d = (static_cast<size_t>(y) * exp_bw + x) * 3;
            for (int c = 0; c < 3; ++c)
                if (source.data[s + c] != boc_img.data[d + c]) crop_matches = false;
        }
    }
    check(crop_matches, "BoC crop pixels match the source region");

    std::filesystem::create_directories("outputs");
    cv::imwrite("outputs/router_boc.png", screen_capture_to_mat(boc_img));
    cv::imwrite("outputs/router_floor.png", screen_capture_to_mat(floor_img));
    std::cout << "Wrote outputs/router_boc.png and outputs/router_floor.png" << std::endl;

    std::cout << (failures ? "router_test FAILED" : "router_test passed") << std::endl;
    return failures ? 1 : 0;
}
