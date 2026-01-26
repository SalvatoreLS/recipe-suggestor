#include "utils.hpp"
#include "types.hpp"
#include <sys/types.h>
#include <cstring>
#include <unordered_map>
#include <string>
#include <fstream>
#include <stdexcept>
#include <nlohmann/json.hpp>

Consumable get_object_from_id(u_int16_t id) {
    // TODO
    return {};
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