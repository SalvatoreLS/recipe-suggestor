#ifndef UTILS_HPP
#define UTILS_HPP

#include <sys/types.h>
#include <string>
#include <opencv2/opencv.hpp>
#include <vector>
#include <unordered_map>
#include "types.hpp"


// Invariant: `data` is always tightly packed BGR24 (3 bytes per pixel, no row
// padding), so its size is always width * height * 3. Every producer must
// convert to this layout; every consumer may rely on it.
struct ScreenCapture {
    unsigned char* data;
    u_int16_t width;
    u_int16_t height;

    ~ScreenCapture() { delete[] data; }
};

// One row of resources/consumables.json (the HASHING.md table).
struct ConsumableInfo {
    types::ConsumableID id = 0;
    std::string name;
    int quality = 0;
    std::string pool_influence;   // human label, e.g. "Devil Room (+10x)"
    std::string pool;             // matching itempools.xml id, e.g. "devil"; empty if none
    int pool_points = 0;          // points this consumable contributes to `pool`
};

// One detector class and the consumables it contributes. A list, because some
// floor labels are quantity multipliers (double_heart) or compound
// (red_soul_heart = one red + one soul).
struct ClassEntry {
    std::string name;
    std::vector<std::pair<types::ConsumableID, types::Quantity>> consumables;
};

// One entry of resources/collectibles.json: the real game data behind a
// craftable item. `quality` is the game's own 0-4 rating and `pools` is the set
// of item pools the collectible can roll from -- an empty `pools` means a
// quest/unlockable item that no pool ever offers.
struct CollectibleInfo {
    types::ItemID id = 0;
    std::string name;
    int quality = 0;
    std::vector<std::string> pools;
    std::vector<std::string> tags;

    bool in_pool(const std::string& pool) const;
};

// Resizes `src` into an `out`-sized canvas while preserving aspect ratio,
// padding the remainder with YOLO's usual gray 114. Ultralytics letterboxes
// during training and validation, so inference must do the same or the model
// sees a distribution it was never trained on. The returned scale/padding are
// what map a model-space box back to source coordinates.
cv::Mat letterbox(const cv::Mat& src, cv::Size out, float& scale, int& pad_x, int& pad_y,
                  const cv::Scalar& pad_colour = cv::Scalar(114, 114, 114));
cv::Mat screen_capture_to_mat(const ScreenCapture& capture);
std::string serializeBag(std::vector<types::ConsumableID>& bag);
std::unordered_map<std::string, types::ItemID> loadJsonToUnorderedMap(const std::string& jsonPath);

// For {"id": "name"} objects such as items.json, whose values are strings
// and so are rejected by loadJsonToUnorderedMap.
std::unordered_map<types::ItemID, std::string> loadJsonToStringMap(const std::string& jsonPath);

// consumables.json is an array of objects, not a flat map.
std::unordered_map<types::ConsumableID, ConsumableInfo> load_consumables(const std::string& jsonPath);

// collectibles.json is a flat {"<id>": {name, quality, pools, tags}} map.
std::unordered_map<types::ItemID, CollectibleInfo> load_collectibles(const std::string& jsonPath);

// Loads one section ("boc" / "floor") of class_map.json. Throws if the section's
// expects_classes does not match the names the loaded model reports -- the
// tripwire for having loaded the wrong .onnx file.
std::vector<ClassEntry> load_class_map(const std::string& jsonPath,
                                       const std::string& section,
                                       const std::vector<std::string>& model_names);

#endif // UTILS_HPP