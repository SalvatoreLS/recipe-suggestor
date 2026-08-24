#include <iostream>
#include <cassert>
#include <vector>
#include <filesystem>
#include "bag_of_crafting.hpp"
#include "types.hpp"

namespace fs = std::filesystem;

// Simple assertion helper
#define ASSERT_EQ(actual, expected) \
    if ((actual) != (expected)) { \
        std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ \
                  << " - Expected: " << (expected) << ", Actual: " << (actual) << std::endl; \
        std::exit(1); \
    }

#define ASSERT_TRUE(condition) \
    if (!(condition)) { \
        std::cerr << "Assertion failed at " << __FILE__ << ":" << __LINE__ \
                  << " - Condition: " << #condition << std::endl; \
        std::exit(1); \
    }

void test_seed_parsing() {
    std::cout << "Testing Seed Parsing..." << std::endl;
    // Known seed and hash (would need to verify MD5 logic manually or assume it works if consistent)
    // "ABCDEFGHI" -> MD5 hash -> lower 32 bits
    // Let's just verify it doesn't crash and returns *something* consistent.
    // Ideally we'd calculate the expected value offline.
    // Empty seed should return 0xd41d8cd9 (md5 of empty) -> lower 32 bits?
    // Actually, parse_seed is private. We can access get_run_seed().
    
    BagOfCrafting boc("test_seed");
    uint32_t seed = boc.get_run_seed();
    std::cout << "Parsed seed: " << seed << std::endl;
    
    BagOfCrafting boc2("test_seed");
    ASSERT_EQ(boc2.get_run_seed(), seed);
}

void test_quality_score() {
    std::cout << "Testing Quality Score..." << std::endl;
    // We can't access calculate_quality_score directly as it's public (Wait, looking at hpp it IS public).
    BagOfCrafting boc("seed");
    
    std::vector<int> ingredients = {1, 2, 3, 4, 0, 1, 2, 3};
    int score = boc.calculate_quality_score(ingredients);
    ASSERT_EQ(score, 16);
    
    ingredients = {4, 4, 4, 4, 4, 4, 4, 4};
    score = boc.calculate_quality_score(ingredients);
    ASSERT_EQ(score, 32);

    ingredients = {};
    score = boc.calculate_quality_score(ingredients);
    ASSERT_EQ(score, 0);
}

void test_quality_range() {
    std::cout << "Testing Quality Range..." << std::endl;
    BagOfCrafting boc("seed");

    // The game's ladder, from the reverse-engineered algorithm. Note the bands
    // OVERLAP -- a mid-range bag can roll anything from quality 2 to 4 -- which
    // the old invented ladder (disjoint, narrowing bands) got wrong.
    struct Case { int score; int lo; int hi; };
    const Case cases[] = {
        {0,  0, 1}, {8,  0, 1},      // <= 8
        {9,  0, 2}, {14, 0, 2},      // 9..14
        {15, 1, 2}, {18, 1, 2},      // 15..18
        {19, 2, 3}, {22, 2, 3},      // 19..22
        {23, 2, 4}, {26, 2, 4},      // 23..26
        {27, 3, 4}, {34, 3, 4},      // 27..34
        {35, 4, 4}, {80, 4, 4},      // > 34
    };
    for (const auto& c : cases) {
        std::pair<int, int> range = boc.get_quality_range(c.score);
        ASSERT_EQ(range.first, c.lo);
        ASSERT_EQ(range.second, c.hi);
    }

    // Devil, Angel and Secret Room pools band 5 points lower, so the same bag
    // draws from a weaker band in those pools.
    // 27 bands as {3,4}; the same 27 in a Devil pool is 22, which bands {2,3}.
    std::pair<int, int> devil = boc.get_quality_range(27, 5);
    ASSERT_EQ(devil.first, 2);
    ASSERT_EQ(devil.second, 3);
}


void test_real_quality_table() {
    std::cout << "Testing real collectible quality table..." << std::endl;
    BagOfCrafting boc("seed");

    // Straight from the game's items_metadata.xml.
    ASSERT_EQ(boc.item_quality(105), 4);   // The D6
    ASSERT_EQ(boc.item_quality(1), 3);     // The Sad Onion
    // Blank/cut ids carry no metadata and are not collectibles.
    ASSERT_EQ(boc.item_quality(43), -1);

    ASSERT_TRUE(!boc.collectibles().empty());
}

void test_pool_points() {
    std::cout << "Testing pool weighting..." << std::endl;
    BagOfCrafting boc("seed");

    // HASHING.md 4: Black Heart is +10 Devil each.
    std::vector<types::ConsumableID> devils(8, 3);
    ASSERT_EQ(boc.pool_points(devils).at("devil"), 80);
    ASSERT_EQ(boc.forced_pool(devils), std::string("devil"));

    // Bone Heart is only +5, so a single one does not reach the threshold.
    std::vector<types::ConsumableID> one_bone = {6, 8, 8, 8, 8, 8, 8, 8};
    ASSERT_EQ(boc.pool_points(one_bone).at("secret"), 5);
    ASSERT_EQ(boc.forced_pool(one_bone), std::string(""));

    // Two reach it.
    std::vector<types::ConsumableID> two_bones = {6, 6, 8, 8, 8, 8, 8, 8};
    ASSERT_EQ(boc.forced_pool(two_bones), std::string("secret"));

    // Plain pennies influence nothing.
    std::vector<types::ConsumableID> pennies(8, 8);
    ASSERT_TRUE(boc.pool_points(pennies).empty());
    ASSERT_EQ(boc.forced_pool(pennies), std::string(""));
}

void test_candidates_are_constrained() {
    std::cout << "Testing candidate constraints..." << std::endl;
    BagOfCrafting boc("seed");

    auto q4 = boc.candidates(4, 4, "");
    ASSERT_TRUE(!q4.empty());
    for (types::ItemID id : q4) ASSERT_EQ(boc.item_quality(id), 4);

    auto devil_q4 = boc.candidates(4, 4, "devil");
    ASSERT_TRUE(!devil_q4.empty());
    ASSERT_TRUE(devil_q4.size() < q4.size());
    for (types::ItemID id : devil_q4) {
        ASSERT_EQ(boc.item_quality(id), 4);
        ASSERT_TRUE(boc.collectibles().at(id).in_pool("devil"));
    }

    // A heuristic craft must respect both constraints. Eight Black Hearts is
    // quality 40 -> band 4-4, and 80 devil points -> the devil pool.
    std::vector<types::ConsumableID> devils(8, 3);
    types::ItemID crafted = boc.craft_item(devils);
    if (boc.lookup_fixed(devils) == 0) {
        ASSERT_EQ(boc.item_quality(crafted), 4);
        ASSERT_TRUE(boc.collectibles().at(crafted).in_pool("devil"));
    }

    // Deterministic for the same seed and bag.
    BagOfCrafting other("seed");
    ASSERT_EQ(other.craft_item(devils), crafted);
}

// E15: the 9/14/18/22/26/30 ladder is still unverified, and fixed.json CANNOT
// verify it -- those recipes are hardcoded exceptions that bypass the quality
// roll entirely, so a mismatch there says nothing about the ladder. This prints
// the comparison for the record without asserting on it; real validation needs
// seed+bag test vectors from the game (TODO G6).
void report_fixed_recipe_quality_bands() {
    std::cout << "Fixed-recipe quality bands (informational, see TODO E15):" << std::endl;
    BagOfCrafting boc("seed");

    int inside = 0, total = 0;
    for (const auto& [bag_key, item_id] : boc.fixed_recipes()) {
        std::vector<types::ConsumableID> bag;
        size_t pos = 0, next;
        std::string key = bag_key;
        while ((next = key.find(',', pos)) != std::string::npos) {
            bag.push_back(static_cast<types::ConsumableID>(std::stoul(key.substr(pos, next - pos))));
            pos = next + 1;
        }
        bag.push_back(static_cast<types::ConsumableID>(std::stoul(key.substr(pos))));

        const auto [lo, hi] = boc.get_quality_range(boc.bag_quality_score(bag));
        int q = boc.item_quality(item_id);
        bool in_band = (q >= lo && q <= hi);
        inside += in_band;
        total++;
        std::cout << "    " << bag_key << " -> " << item_id
                  << " q=" << q << " band=" << lo << "-" << hi
                  << (in_band ? "  in" : "  OUT") << std::endl;
    }
    std::cout << "  ladder agrees with " << inside << "/" << total
              << " hardcoded recipes (not a pass/fail signal)" << std::endl;
}

int main() {
    try {
        test_seed_parsing();
        test_quality_score();
        test_quality_range();
        test_real_quality_table();
        test_pool_points();
        test_candidates_are_constrained();
        report_fixed_recipe_quality_bands();
        
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
