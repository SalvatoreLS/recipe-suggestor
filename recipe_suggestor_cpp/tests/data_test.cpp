#include <iostream>
#include <string>
#include "utils.hpp"
#include "constants.hpp"

static int failures = 0;

static void check(bool cond, const std::string& what) {
    std::cout << (cond ? "  ok   " : "  FAIL ") << what << std::endl;
    if (!cond) failures++;
}

int main() {
    try {
        auto consumables = load_consumables(constants::consumables_path);
        check(consumables.size() == 29, "consumables.json has 29 entries");
        check(consumables.at(1).name == "Red Heart" && consumables.at(1).quality == 1,
              "consumable 1 is Red Heart with quality 1");
        check(consumables.at(11).quality == 8, "Lucky Penny has quality 8");
        check(consumables.at(29).quality == 0, "Poop Nugget has quality 0");

        auto items = loadJsonToStringMap(constants::item_names_path);
        check(items.size() == 726, "items.json has 726 entries (got " + std::to_string(items.size()) + ")");
        check(items.at(45) == "Yum Heart", "item 45 is Yum Heart");
        check(items.at(686) == "Soul Locket", "item 686 is Soul Locket");

        // Every fixed recipe must resolve to a real collectible name.
        auto fixed = loadJsonToUnorderedMap(constants::fixed_crafts_path);
        check(!fixed.empty(), "fixed.json is not empty (got " + std::to_string(fixed.size()) + " recipes)");
        bool all_resolve = true;
        for (const auto& [bag, item_id] : fixed) {
            if (items.find(item_id) == items.end()) {
                std::cout << "    unresolved item id " << item_id << " for bag " << bag << std::endl;
                all_resolve = false;
            }
        }
        check(all_resolve, "every fixed.json result resolves to an item name");

        // Class maps. Passing an empty model_names skips the tripwire, which is
        // what we want here: this test must not require the .onnx files.
        const std::pair<std::string, size_t> sections[] = {{"boc", 21}, {"floor", 23}};
        for (const auto& [section, expected] : sections) {
            auto map = load_class_map(constants::class_map_path, section, {});
            check(map.size() == expected,
                  section + " class map has " + std::to_string(expected) + " entries (got " +
                  std::to_string(map.size()) + ")");

            bool refs_ok = true;
            for (const auto& entry : map) {
                for (const auto& [cid, qty] : entry.consumables) {
                    if (consumables.find(cid) == consumables.end() || qty == 0) {
                        std::cout << "    bad mapping in " << section << ": " << entry.name << std::endl;
                        refs_ok = false;
                    }
                }
            }
            check(refs_ok, section + " maps only to known consumables with non-zero qty");
        }

        auto boc = load_class_map(constants::class_map_path, "boc", {});
        check(boc[7].name == "extra" && boc[7].consumables.empty(), "boc 'extra' maps to nothing");

        auto floor = load_class_map(constants::class_map_path, "floor", {});
        check(floor[7].name == "double_heart" && floor[7].consumables.size() == 1 &&
              floor[7].consumables[0].second == 4, "double_heart is 4x Red Heart");
        check(floor[20].name == "red_soul_heart" && floor[20].consumables.size() == 2,
              "red_soul_heart maps to two different consumables");
        check(floor[1].name == "black_heart" && floor[15].name == "micro_battery",
              "floor map carries the two classes the v2 dataset added");

        // The real game data extracted by models_training/extract_game_data.py.
        auto collectibles = load_collectibles(constants::collectibles_path);
        check(collectibles.size() == 721,
              "collectibles.json has 721 entries (got " + std::to_string(collectibles.size()) + ")");

        bool quality_in_range = true, names_agree = true;
        for (const auto& [id, info] : collectibles) {
            if (info.quality < 0 || info.quality > 4) quality_in_range = false;
            auto it = items.find(id);
            if (it == items.end() || it->second != info.name) names_agree = false;
        }
        check(quality_in_range, "every collectible quality is 0-4");
        check(names_agree, "collectible names agree with items.json");

        // Spot checks against the game's own table.
        check(collectibles.at(105).name == "The D6" && collectibles.at(105).quality == 4,
              "The D6 is quality 4");
        check(collectibles.at(105).in_pool("treasure"), "The D6 is in the treasure pool");

        // The blank/cut/unused ids in items.json carry no metadata and must not
        // be craftable -- the old code would happily have crafted item 43 (\"\").
        for (types::ItemID cut : {43, 61, 235, 587, 718}) {
            check(collectibles.find(cut) == collectibles.end(),
                  "item " + std::to_string(cut) + " is excluded as non-craftable");
        }

        // Every fixed recipe result must be a real, craftable collectible.
        bool fixed_craftable = true;
        for (const auto& [bag, item_id] : fixed) {
            if (collectibles.find(item_id) == collectibles.end()) {
                std::cout << "    fixed.json result " << item_id << " is not a collectible" << std::endl;
                fixed_craftable = false;
            }
        }
        check(fixed_craftable, "every fixed.json result is a craftable collectible");

        // The tripwire itself must fire on a mismatched class list.
        bool threw = false;
        try {
            load_class_map(constants::class_map_path, "boc", {"not", "the", "right", "classes"});
        } catch (const std::exception&) {
            threw = true;
        }
        check(threw, "wrong model class list is rejected");

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << (failures ? "data_test FAILED" : "data_test passed") << std::endl;
    return failures ? 1 : 0;
}
