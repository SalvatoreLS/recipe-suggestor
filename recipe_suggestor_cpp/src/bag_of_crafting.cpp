#include "bag_of_crafting.hpp"
#include <openssl/md5.h>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include "utils.hpp"
#include "constants.hpp"

BagOfCrafting::BagOfCrafting(const std::string& run_seed) 
    : run_seed_(parse_seed(run_seed)) {
        this->fixed_crafts = loadJsonToUnorderedMap(constants::fixed_crafts_path);
}

uint32_t BagOfCrafting::parse_seed(const std::string& seed_str) const {
    // Compute MD5 hash
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char*>(seed_str.c_str()), 
        seed_str.length(), 
        digest);
    
    // Use first 4 bytes of digest as seed
    uint32_t seed;
    // Assuming 4 bytes from digest are sufficient
    // Copy bytes 0-3 to seed
    std::memcpy(&seed, digest, sizeof(uint32_t));
    return seed;
}

int BagOfCrafting::calculate_quality_score(const std::vector<int>& ingredients) const {
    int sum = 0;
    for (int quality : ingredients) {
        sum += quality;
    }
    return sum;
}

std::pair<int, int> BagOfCrafting::get_quality_range(int score) const {
    if (score < 9)        return {0, 0};
    if (score <= 14)      return {0, 1};
    if (score <= 18)      return {1, 2};
    if (score <= 22)      return {2, 3};
    if (score <= 26)      return {3, 3};
    if (score <= 30)      return {3, 4};
    return {4, 4};
}

types::ItemID BagOfCrafting::craft_item(const std::vector<types::ConsumableID>& ingredient_ids) {
    // 1. Sort IDs to ensure order doesn't change outcome
    std::vector<types::ConsumableID> sorted_ids = ingredient_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());

    std::string serial_bag = serializeBag(sorted_ids);
    if (this->fixed_crafts.find(serial_bag) != this->fixed_crafts.end()) return this->fixed_crafts[serial_bag];
    
    // 2. Initialize the internal PRNG with seed and components
    // This mimics the game's internal hashing
    uint32_t h = run_seed_;
    for (types::ConsumableID item_id : sorted_ids) {
        h = (h * 1103515245 + 12345) & 0x7FFFFFFF;
        h = (h ^ item_id) & 0x7FFFFFFF;
    }
    
    // 3. Determine quality constraints
    // Calculate qualities of ingredients (would come from consumable data table)
    std::vector<int> qualities;
    qualities.reserve(sorted_ids.size());
    for (types::ConsumableID id : sorted_ids) {
        // In actual implementation, look up quality from consumable data
        // For now, using a deterministic mapping based on ID
        qualities.push_back((id % 5)); // Placeholder: maps to 0-4 quality
    }
    
    int total_quality = calculate_quality_score(qualities);
    std::pair<int, int> target_range = get_quality_range(total_quality);
    
    // 4. Iterative hashing to find a valid Item ID
    types::ItemID crafted_item_id = find_matching_item(h, target_range);
    return crafted_item_id;
}

types::ItemID BagOfCrafting::find_matching_item(uint32_t current_hash, 
                                                const std::pair<int, int>& quality_range) const {
    // The game uses a pool of 719 craftable items (item IDs 1-719 in the original pool)
    // It iterates through RNG states until finding a valid item
    constexpr uint32_t POOL_SIZE = 719;
    constexpr uint32_t MAX_ATTEMPTS = 10000; // Prevent infinite loops
    
    uint32_t h = current_hash;
    
    for (uint32_t attempt = 0; attempt < MAX_ATTEMPTS; ++attempt) {
        // Calculate candidate item ID from current hash
        types::ItemID candidate_id = (h % POOL_SIZE) + 1;
        
        // Check if item is craftable and within quality range
        if (is_craftable_item(candidate_id)) {
            int item_quality = get_item_quality(candidate_id);
            
            if (item_quality >= quality_range.first && 
                item_quality <= quality_range.second) {
                return candidate_id;
            }
        }
        
        // Advance RNG state and try again
        h = next_prng(h);
    }
    
    // Fallback: return the raw modulo result if no valid item found
    return (current_hash % POOL_SIZE) + 1;
}

uint32_t BagOfCrafting::next_prng(uint32_t state) const {
    // Linear congruential generator step (same as used in the hashing)
    return ((state * 1103515245 + 12345) & 0x7FFFFFFF);
}

int BagOfCrafting::get_item_quality(types::ItemID item_id) const {
    // In the actual game, this would look up the item in a data table
    // Isaac items have quality ratings from 0-4
    // This is a placeholder that should be replaced with actual item data
    
    // Example mapping (would be loaded from game data):
    // Quality 0: Common items
    // Quality 1: Decent items
    // Quality 2: Good items
    // Quality 3: Great items
    // Quality 4: Special/Best items
    
    // Placeholder: deterministic quality based on item ID
    // In production, replace with actual item quality lookup
    if (item_id <= 143) return 0;
    if (item_id <= 286) return 1;
    if (item_id <= 429) return 2;
    if (item_id <= 572) return 3;
    return 4;
}

bool BagOfCrafting::is_craftable_item(types::ItemID item_id) const {
    // In the actual game, this checks if the item is in the craftable pool
    // Some items are excluded from crafting (e.g., quest items, story items)
    
    // TODO: check how this function can be implemented

    // The base craftable pool is item IDs 1-719 with some exclusions
    // This is a placeholder that should be replaced with actual game data
    
    if (item_id < 1 || item_id > 719) {
        return false;
    }
    
    // Example exclusions (would be loaded from game data):
    // Exclude certain quest items or special items
    // For now, accept all items in range 1-719
    return true;
}