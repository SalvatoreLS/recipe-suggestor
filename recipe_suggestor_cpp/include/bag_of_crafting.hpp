#ifndef BAG_OF_CRAFTING_HPP
#define BAG_OF_CRAFTING_HPP

#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include "types.hpp"
#include "utils.hpp"


class BagOfCrafting {
public:
    explicit BagOfCrafting(const std::string& run_seed);
    
    // Calculate total quality score from ingredients
    int calculate_quality_score(const std::vector<int>& ingredients) const;
    
    // Get quality range based on score
    std::pair<int, int> get_quality_range(int score) const;
    
    // Craft an item from ingredient IDs
    types::ItemID craft_item(const std::vector<types::ConsumableID>& ingredient_ids);
    
    // Get the parsed run seed
    uint32_t get_run_seed() const { return run_seed_; }

private:
    uint32_t run_seed_;
    std::unordered_map<std::string, types::ItemID> fixed_crafts;

    // Parse seed string to unsigned 32-bit integer using MD5
    uint32_t parse_seed(const std::string& seed_str) const;
    
    // Find matching item ID based on hash and quality constraints
    types::ItemID find_matching_item(uint32_t current_hash, 
                                     const std::pair<int, int>& quality_range) const;
    
    // PRNG step function used by the game
    uint32_t next_prng(uint32_t state) const;
    
    // Get item quality (would typically come from item data table)
    int get_item_quality(types::ItemID item_id) const;
    
    // Check if item is in the craftable pool
    bool is_craftable_item(types::ItemID item_id) const;
};

#endif // BAG_OF_CRAFTING_HPP