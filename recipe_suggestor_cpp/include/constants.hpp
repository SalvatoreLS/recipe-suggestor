#pragma once

#include <sys/types.h>

namespace constants {

    // Detector constants
    inline constexpr float default_conf_threshold = 0.5f;
    inline constexpr float nms_threshold = 0.1f;
    inline constexpr int sort_threshold = 20;
    inline constexpr float pixel_scale = 1.0f / 255.0f;
    inline constexpr int intra_op_num_threads = 1;

    // Router constants
    inline constexpr double boc_crop_width_factor = 0.35;
    inline constexpr double boc_crop_height_factor = 0.18;
    inline constexpr double crop_start_x_factor = 0.97;
    inline constexpr double left_section_x_factor = 0.0;
    inline constexpr double left_section_y_factor = 0.13;
    inline constexpr double left_crop_width_factor = 0.13;
    inline constexpr double left_crop_height_factor = 0.7;

    // Trie constants
    inline constexpr int max_elements_rank = 15;

    // Processing Queue
    inline constexpr u_int16_t queue_max_size = 10;
}