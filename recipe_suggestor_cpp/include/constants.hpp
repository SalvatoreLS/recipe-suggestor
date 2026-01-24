#pragma once

namespace constants {

    // FloorDetector constants
    // TODO

    // BoCDetector constants
    // TODO

    // TODO: define two sets of thresholds for the two detectors (different requirements)

    // Detector constants
    inline constexpr float default_conf_threshold = 0.5f;
    inline constexpr float nms_threshold = 0.1f;
    inline constexpr int sort_threshold = 20;
    inline constexpr float pixel_scale = 1.0f / 255.0f;
    inline constexpr int intra_op_num_threads = 1;

    // Router constants
    inline constexpr double crop_width_factor = 0.35;
    inline constexpr double crop_height_factor = 0.18;
    inline constexpr double crop_start_x_factor = 0.97;

    // Trie constants
    inline constexpr int max_elements_rank = 15;
}
