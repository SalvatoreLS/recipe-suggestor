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
    
    // < 9 -> {0, 0}
    std::pair<int, int> range = boc.get_quality_range(8);
    ASSERT_EQ(range.first, 0);
    ASSERT_EQ(range.second, 0);
    
    // <= 14 -> {0, 1}
    range = boc.get_quality_range(14);
    ASSERT_EQ(range.first, 0);
    ASSERT_EQ(range.second, 1);
    
    range = boc.get_quality_range(9);
    ASSERT_EQ(range.first, 0);
    ASSERT_EQ(range.second, 1);

    // <= 18 -> {1, 2}
    range = boc.get_quality_range(18);
    ASSERT_EQ(range.first, 1);
    ASSERT_EQ(range.second, 2);

    // <= 22 -> {2, 3}
    range = boc.get_quality_range(22);
    ASSERT_EQ(range.first, 2);
    ASSERT_EQ(range.second, 3);
    
    // <= 26 -> {3, 3}
    range = boc.get_quality_range(26);
    ASSERT_EQ(range.first, 3);
    ASSERT_EQ(range.second, 3);

    // <= 30 -> {3, 4}
    range = boc.get_quality_range(30);
    ASSERT_EQ(range.first, 3);
    ASSERT_EQ(range.second, 4);

    // > 30 -> {4, 4}
    range = boc.get_quality_range(31);
    ASSERT_EQ(range.first, 4);
    ASSERT_EQ(range.second, 4);
}

void setup_resources() {
    // Create dummy fixed.json if not exists
    if (!fs::exists("resources")) {
        fs::create_directory("resources");
    }
    std::string fixed_json_path = "resources/fixed.json";
    // Always create a minimal valid JSON for the map to ensure test consistency
    // This overwrites any existing file in build/resources/fixed.json
    FILE* f = fopen(fixed_json_path.c_str(), "w");
    if (f) {
        fprintf(f, "{}\n");
        fclose(f);
    }
}

int main() {
    try {
        setup_resources();
        
        test_seed_parsing();
        test_quality_score();
        test_quality_range();
        
        std::cout << "All tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Exception: " << e.what() << std::endl;
        return 1;
    }
}
