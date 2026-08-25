#pragma once

#include <sys/types.h>
#include <string>

namespace constants {

    // Detector constants
    inline constexpr float default_conf_threshold = 0.5f;
    // The floor model is trained on far less data, so it needs a lower gate.
    inline constexpr float floor_conf_threshold = 0.35f;
    // IoU above which overlapping boxes are suppressed. 0.1 collapsed anything
    // overlapping by >10%, which wrecked the tightly packed 8-slot bag row.
    inline constexpr float nms_threshold = 0.45f;
    inline constexpr int sort_threshold = 20;
    inline constexpr float pixel_scale = 1.0f / 255.0f;
    inline constexpr int intra_op_num_threads = 1;

    // Router constants
    // Bag strip geometry, calibrated against a real 1920x1080 frame. Two
    // independent measurements agree: template-matching the bag sprite from a
    // BoC training crop onto a live frame (match 0.96) puts the strip at
    // (1189, 885) 730x194, and the black bag mask baked into every floor
    // training image sits at (1191, 888) 729x192. The strip therefore ends at
    // the right screen edge, and its 730:194 aspect is exactly the 640:170
    // content band of the BoC training images -- which is what makes
    // LetterboxBlack reproduce the training layout pixel for pixel.
    inline constexpr double boc_crop_width_factor = 0.38;
    inline constexpr double boc_crop_height_factor = 0.18;
    inline constexpr double crop_start_x_factor = 1.0;
    inline constexpr double left_section_x_factor = 0.0;
    inline constexpr double left_section_y_factor = 0.13;
    inline constexpr double left_crop_width_factor = 0.13;
    inline constexpr double left_crop_height_factor = 0.7;

    // How many alternative plans to print under the recommended one. The list
    // is ranked, so more than a few is noise while playing.
    inline constexpr size_t max_shown_alternatives = 3;

    // Trie constants
    inline constexpr int max_elements_rank = 15;

    // Processing Queue
    inline constexpr u_int16_t queue_max_size = 10;

    // Preprocessing geometry must match how each model was TRAINED, not what is
    // generally "better", and the two datasets do NOT agree -- so the mode is a
    // per-detector property (types::Preprocess), not one global flag:
    //   floor -- whole frames exported from Roboflow with "Resize to 640x640
    //            (Stretch)". Letterboxing them instead cost 18 points of recall
    //            on the validation set (68.6% vs 86.9% of ground-truth
    //            instances), which is what produced the old global flag.
    //   BoC   -- a wide bag strip black-padded to a square before Roboflow ever
    //            saw it: content occupies rows 235..405 of every training image,
    //            i.e. a 640x170 band centred in 640x640. Squashing the live
    //            730x194 strip to 640x640 stretched sprites 3.8x vertically
    //            against what the model learned.

    // Image constants
    inline constexpr u_int16_t img_width = 640;
    inline constexpr u_int16_t img_height = 640;

    // Fixed paths
    inline constexpr const char* fixed_crafts_path = "resources/fixed.json";
    inline constexpr const char* boc_model_path = "resources/models/boc_best.onnx";
    inline constexpr const char* floor_model_path = "resources/models/floor_best.onnx";
    inline constexpr const char* consumables_path = "resources/consumables.json";
    inline constexpr const char* class_map_path = "resources/class_map.json";
    // The complete 726-entry collectible table (id -> name).
    inline constexpr const char* item_names_path = "resources/items.json";
    // Real per-collectible quality and pool membership, extracted from the game
    // by models_training/extract_game_data.py. 721 entries: the handful of blank,
    // cut and unused ids in items.json carry no metadata and are not craftable.
    inline constexpr const char* collectibles_path = "resources/collectibles.json";
    // The reverse-engineered crafting RNG tables (per-component xorshift shifts
    // and pickup values) and the game's pools in file order with item weights.
    // Crafting indexes pools by position, so the order of itempools.json matters.
    inline constexpr const char* crafting_rng_path = "resources/crafting_rng.json";
    inline constexpr const char* itempools_path = "resources/itempools.json";

    // Crafting
    inline constexpr size_t bag_size = 8;
    inline constexpr size_t max_suggestion_candidates = 20000;
}