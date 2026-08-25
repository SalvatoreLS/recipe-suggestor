#include <algorithm>
#include <iostream>
#include <string>
#include "pipeline/nodes/recipe_suggestor.hpp"

static int failures = 0;

static void check(bool cond, const std::string& what) {
    std::cout << (cond ? "  ok   " : "  FAIL ") << what << std::endl;
    if (!cond) failures++;
}

// Class indices in the BoC / floor sections of class_map.json.
namespace boc_cls {
constexpr types::ConsumableID extra = 7;
constexpr types::ConsumableID penny = 15;
constexpr types::ConsumableID red_heart = 17;
}
// The v2 floor dataset added black_heart (1) and micro_battery (15), shifting
// every index after them.
namespace floor_cls {
constexpr types::ConsumableID double_heart = 7;
constexpr types::ConsumableID penny = 17;
constexpr types::ConsumableID soul_heart = 22;
}

// Consumable ids from consumables.json.
constexpr types::ConsumableID PENNY = 8;
constexpr types::ConsumableID RED_HEART = 1;
constexpr types::ConsumableID SOUL_HEART = 2;

int main() {
    try {
        RecipeSuggestor suggestor("test seed");

        // --- translation -------------------------------------------------
        auto bag = suggestor.translate_bag({boc_cls::penny, boc_cls::penny, boc_cls::extra});
        check(bag.size() == 2 && bag[0] == PENNY && bag[1] == PENNY,
              "'extra' is dropped, pennies translate to consumable 8");

        auto supply = suggestor.translate_floor({{floor_cls::double_heart, 1}});
        check(supply.size() == 1 && supply[RED_HEART] == 4, "double_heart yields 4x Red Heart");

        auto supply2 = suggestor.translate_floor({{floor_cls::penny, 3}});
        check(supply2[PENNY] == 3, "three pennies on the floor yield 3 units");

        auto over = suggestor.translate_bag(std::vector<types::ConsumableID>(12, boc_cls::red_heart));
        check(over.size() == constants::bag_size, "bag is truncated to 8 even with spurious detections");

        // --- a full bag that is a known fixed recipe ---------------------
        std::vector<types::ConsumableID> eight_pennies(8, boc_cls::penny);
        auto full = suggestor.suggest(eight_pennies, {});
        check(!full.empty(), "a full bag of 8 pennies produces a suggestion");
        if (!full.empty()) {
            check(full[0].exact, "8 pennies is an exact (fixed.json) recipe");
            check(full[0].item_id == 177, "8 pennies craft item #177 (got " + std::to_string(full[0].item_id) + ")");
            check(full[0].to_add.empty(), "a full bag needs nothing added");
            check(full[0].quality_score == 8, "8 pennies score 8 (quality 1 each)");
            check(!full[0].item_name.empty() && full[0].item_name[0] != 'I',
                  "the crafted item resolves to a real name: " + full[0].item_name);
        }

        // --- 7 in the bag, the eighth on the floor -----------------------
        std::vector<types::ConsumableID> seven_pennies(7, boc_cls::penny);
        auto partial = suggestor.suggest(seven_pennies, {{floor_cls::penny, 1}});
        check(!partial.empty(), "7 pennies + a penny on the floor produces a suggestion");
        if (!partial.empty()) {
            check(partial[0].exact && partial[0].item_id == 177, "the same fixed recipe surfaces");
            check(partial[0].to_add.size() == 1 && partial[0].to_add[0] == PENNY,
                  "it tells you to pick up one penny");
        }

        // --- the memo cache ---------------------------------------------
        size_t before = suggestor.computations();
        suggestor.suggest(seven_pennies, {{floor_cls::penny, 1}});
        suggestor.suggest(seven_pennies, {{floor_cls::penny, 1}});
        check(suggestor.computations() == before, "identical inputs are served from the memo cache");

        suggestor.suggest(eight_pennies, {});
        check(suggestor.computations() == before + 1, "changed inputs recompute exactly once");

        // --- naming is gated on knowing the run's crafting seed ------------
        std::vector<types::ConsumableID> odd_bag = {boc_cls::penny, boc_cls::penny, boc_cls::penny,
                                                    boc_cls::penny, boc_cls::penny, boc_cls::penny,
                                                    boc_cls::penny, boc_cls::red_heart};
        auto guarded = suggestor.suggest(odd_bag, {});
        // Without the start seed the ITEM is unknowable, but the plan and its
        // quality band are real game data and must still be reported -- dropping
        // the whole entry is what left the console silent on a full bag.
        check(!guarded.empty(), "an unknown bag still yields a plan without the start seed");
        if (!guarded.empty()) {
            check(guarded[0].item_id == 0, "the unknown item is reported as no item, not invented");
            check(!guarded[0].exact, "it is not claimed to be exact");
            check(guarded[0].recipe.size() == constants::bag_size, "the plan is a full bag of 8");
            check(guarded[0].quality_score == 7 * 1 + 1, "the real quality score is still computed (7 pennies + a red heart)");
            check(guarded[0].band_hi >= guarded[0].band_lo, "a quality band is attached");
        }

        // With a start seed the same bag resolves to a concrete collectible,
        // computed by the game's own algorithm.
        suggestor.set_start_seed(0xDEADBEEF);
        auto seeded = suggestor.suggest(odd_bag, {});
        check(!seeded.empty(), "the same bag still yields a plan with a start seed");
        if (!seeded.empty()) {
            check(seeded[0].item_id != 0, "the crafted item is now named");
            check(!seeded[0].exact, "a seed-computed item is not marked as a fixed recipe");
            check(seeded[0].recipe.size() == constants::bag_size, "the recipe is a full bag of 8");
        }

        // --- a full bag is still plannable -------------------------------
        // Picking up past a full bag pushes the oldest entries out, so a full
        // bag has options; before this it produced nothing at all.
        auto full_bag_plans = suggestor.suggest(eight_pennies, {{floor_cls::soul_heart, 1}});
        check(!full_bag_plans.empty(), "a full bag still yields plans when the floor has pickups");
        bool found_swap = false, asks_for_half_a_pickup = false;
        for (const auto& p : full_bag_plans) {
            if (p.pickups.size() == 1 && p.to_drop.size() == 2) found_swap = true;
            // A Soul Heart on the floor is two bag units. A plan may take both
            // or neither -- never one, which would be half a pickup.
            int soul_units = 0;
            for (types::ConsumableID id : p.to_add) if (id == SOUL_HEART) soul_units++;
            if (soul_units % 2 != 0) asks_for_half_a_pickup = true;
        }
        check(found_swap, "one Soul Heart pickup swaps out the two oldest pennies");
        check(!asks_for_half_a_pickup, "no plan asks for a fraction of an indivisible pickup");
        // Guaranteed (fixed.json) recipes come first because their item is
        // actually known; quality orders everything within each group.
        bool ranked = true;
        for (size_t i = 1; i < full_bag_plans.size(); ++i) {
            const auto& prev = full_bag_plans[i - 1];
            const auto& cur = full_bag_plans[i];
            if (prev.exact != cur.exact) { if (!prev.exact) ranked = false; continue; }
            if (cur.quality_score > prev.quality_score) ranked = false;
        }
        check(ranked, "plans are ranked: guaranteed recipes first, then by quality");

        // --- formatting ---------------------------------------------------
        std::string text = suggestor.format(partial);
        check(text.find("Penny") != std::string::npos, "output names consumables in plain English");
        check(text.find('*') != std::string::npos, "exact recipes are starred");
        std::cout << "\n--- sample output ---\n" << text << "---------------------\n";

    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }

    std::cout << (failures ? "suggestor_test FAILED" : "suggestor_test passed") << std::endl;
    return failures ? 1 : 0;
}
