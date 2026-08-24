#include "utils.hpp"
#include "types.hpp"
#include <sys/types.h>
#include <algorithm>
#include <cstring>
#include <unordered_map>
#include <string>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

cv::Mat letterbox(const cv::Mat& src, cv::Size out, float& scale, int& pad_x, int& pad_y,
                  const cv::Scalar& pad_colour) {
    scale = std::min(static_cast<float>(out.width) / src.cols,
                     static_cast<float>(out.height) / src.rows);

    int new_w = static_cast<int>(std::round(src.cols * scale));
    int new_h = static_cast<int>(std::round(src.rows * scale));

    pad_x = (out.width - new_w) / 2;
    pad_y = (out.height - new_h) / 2;

    cv::Mat resized;
    cv::resize(src, resized, cv::Size(new_w, new_h), 0, 0, cv::INTER_LINEAR);

    cv::Mat canvas(out, src.type(), pad_colour);
    resized.copyTo(canvas(cv::Rect(pad_x, pad_y, new_w, new_h)));
    return canvas;
}

cv::Mat screen_capture_to_mat(const ScreenCapture& capture) {
    cv::Mat mat(capture.height, capture.width, CV_8UC3);    
    std::memcpy(mat.data, capture.data, capture.width * capture.height * 3);
    
    return mat;
}

    std::string serializeBag(std::vector<types::ConsumableID>& bag) {
        if (bag.empty()) return "";

        // Sort to ensure the key is deterministic regardless of pickup order
        std::sort(bag.begin(), bag.end());

        std::stringstream ss;
        for (size_t i = 0; i < bag.size(); ++i) {
            ss << bag[i];
            if (i < bag.size() - 1) {
                ss << ",";
            }
        }

        return ss.str();
    }

std::unordered_map<std::string, types::ItemID> loadJsonToUnorderedMap(const std::string& jsonPath) {
    std::ifstream in(jsonPath);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + jsonPath);
    }

    nlohmann::json j;
    in >> j;

    if (!j.is_object()) {
        throw std::runtime_error("JSON root is not an object");
    }

    std::unordered_map<std::string, types::ItemID> result;
    result.reserve(j.size());

    for (const auto& [key, value] : j.items()) {
        if (!value.is_number_integer()) {
            throw std::runtime_error("Non-integer value for key: " + key);
        }
        result.emplace(key, value.get<types::ItemID>());
    }

    return result;
}
std::unordered_map<types::ItemID, std::string> loadJsonToStringMap(const std::string& jsonPath) {
    std::ifstream in(jsonPath);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + jsonPath);
    }

    nlohmann::json j;
    in >> j;

    if (!j.is_object()) {
        throw std::runtime_error("JSON root is not an object: " + jsonPath);
    }

    std::unordered_map<types::ItemID, std::string> result;
    result.reserve(j.size());

    for (const auto& [key, value] : j.items()) {
        if (!value.is_string()) continue;
        result.emplace(static_cast<types::ItemID>(std::stoi(key)), value.get<std::string>());
    }

    return result;
}

std::unordered_map<types::ConsumableID, ConsumableInfo> load_consumables(const std::string& jsonPath) {
    std::ifstream in(jsonPath);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + jsonPath);
    }

    nlohmann::json j;
    in >> j;

    if (!j.is_array()) {
        throw std::runtime_error("consumables.json root is not an array: " + jsonPath);
    }

    std::unordered_map<types::ConsumableID, ConsumableInfo> result;
    for (const auto& entry : j) {
        ConsumableInfo info;
        info.id = entry.at("id").get<types::ConsumableID>();
        info.name = entry.at("consumable").get<std::string>();
        info.quality = entry.at("quality").get<int>();
        if (entry.contains("pool_influence") && entry.at("pool_influence").is_string()) {
            info.pool_influence = entry.at("pool_influence").get<std::string>();
        }
        if (entry.contains("pool") && entry.at("pool").is_string()) {
            info.pool = entry.at("pool").get<std::string>();
        }
        if (entry.contains("pool_points")) {
            info.pool_points = entry.at("pool_points").get<int>();
        }
        result.emplace(info.id, std::move(info));
    }

    return result;
}

bool CollectibleInfo::in_pool(const std::string& pool) const {
    return std::find(pools.begin(), pools.end(), pool) != pools.end();
}

std::unordered_map<types::ItemID, CollectibleInfo> load_collectibles(const std::string& jsonPath) {
    std::ifstream in(jsonPath);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + jsonPath);
    }

    nlohmann::json j;
    in >> j;

    if (!j.is_object()) {
        throw std::runtime_error("collectibles.json root is not an object: " + jsonPath);
    }

    std::unordered_map<types::ItemID, CollectibleInfo> result;
    for (const auto& [key, entry] : j.items()) {
        CollectibleInfo info;
        info.id = static_cast<types::ItemID>(std::stoul(key));
        info.name = entry.at("name").get<std::string>();
        info.quality = entry.at("quality").get<int>();

        // A collectible's quality is the game's own 0-4 rating; anything else
        // means the extraction went wrong and every quality gate downstream
        // would be silently meaningless.
        if (info.quality < 0 || info.quality > 4) {
            throw std::runtime_error("collectibles.json: item " + key +
                                     " has out-of-range quality " +
                                     std::to_string(info.quality));
        }

        if (entry.contains("pools")) {
            for (const auto& pool : entry.at("pools")) {
                info.pools.push_back(pool.get<std::string>());
            }
        }
        if (entry.contains("tags")) {
            for (const auto& tag : entry.at("tags")) {
                info.tags.push_back(tag.get<std::string>());
            }
        }
        result.emplace(info.id, std::move(info));
    }

    return result;
}

std::vector<ClassEntry> load_class_map(const std::string& jsonPath,
                                       const std::string& section,
                                       const std::vector<std::string>& model_names) {
    std::ifstream in(jsonPath);
    if (!in.is_open()) {
        throw std::runtime_error("Failed to open JSON file: " + jsonPath);
    }

    nlohmann::json j;
    in >> j;

    if (!j.contains(section)) {
        throw std::runtime_error("class_map.json has no section '" + section + "'");
    }
    const auto& sec = j.at(section);

    std::vector<std::string> expected = sec.at("expects_classes").get<std::vector<std::string>>();
    if (!model_names.empty() && model_names != expected) {
        throw std::runtime_error(
            "Model class list does not match class_map.json section '" + section +
            "'. The wrong .onnx file is almost certainly loaded: both models are nc:21 "
            "and produce an identical output shape.");
    }

    std::vector<ClassEntry> result;
    for (const auto& item : sec.at("map")) {
        ClassEntry e;
        e.name = item.at("name").get<std::string>();
        for (const auto& c : item.at("consumables")) {
            e.consumables.emplace_back(c.at("id").get<types::ConsumableID>(),
                                       c.at("qty").get<types::Quantity>());
        }
        result.push_back(std::move(e));
    }

    if (result.size() != expected.size()) {
        throw std::runtime_error("class_map.json section '" + section + "': map/expects_classes size mismatch");
    }

    return result;
}
