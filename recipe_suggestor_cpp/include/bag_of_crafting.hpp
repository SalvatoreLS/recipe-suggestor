#ifndef BAG_OF_CRAFTING_HPP
#define BAG_OF_CRAFTING_HPP

#include <array>
#include <functional>
#include <string>
#include <vector>
#include <cstdint>
#include <unordered_map>
#include "types.hpp"
#include "utils.hpp"
#include "constants.hpp"

// The real Repentance crafting algorithm.
//
// craft_item() is a port of the reverse-engineered routine used by External Item
// Descriptions (wofsauge/External-Item-Descriptions, features/eid_bagofcrafting.lua,
// EID:calculateBagOfCrafting): the ingredients are sorted by id, each one shifts a
// xorshift RNG seeded with the run's START SEED, the bag's total pickup value picks a
// quality band per pool, every pool contributes its items weighted by pool weight x item
// weight, and up to 20 rolls draw from that distribution.
//
// The data it runs on is the game's own: resources/crafting_rng.json (the per-component
// shift triples and pickup values), resources/itempools.json (the pools in file order with
// per-item weights), resources/collectibles.json (quality), resources/fixed.json (the
// hardcoded recipes, checked first).
//
// The one thing that is NOT derivable from the seed STRING is the 32-bit start seed the
// RNG needs -- the game shows the string, the algorithm needs the number, and the encoding
// between them is not public. set_start_seed() takes it directly; find_start_seed()
// recovers it from observed (bag -> item) crafts. Until it is set, craft_item() can only
// answer for fixed recipes.
class BagOfCrafting {
public:
    explicit BagOfCrafting(const std::string& run_seed,
                           const std::string& fixed_path = constants::fixed_crafts_path,
                           const std::string& consumables_path = constants::consumables_path,
                           const std::string& items_path = constants::item_names_path,
                           const std::string& collectibles_path = constants::collectibles_path);

    // Sum of the real consumable qualities (resources/consumables.json,
    // matching the HASHING.md table).
    int bag_quality_score(const std::vector<types::ConsumableID>& bag) const;

    // Kept for the quality ladder itself, which is independent of where the
    // per-item numbers come from.
    int calculate_quality_score(const std::vector<int>& qualities) const;
    // The game's quality band for a bag total. `pool_penalty` is the 5 points the
    // Devil, Angel and Secret Room pools subtract before banding, so the same bag
    // can draw from a different band depending on which pool is being considered.
    std::pair<int, int> get_quality_range(int score, int pool_penalty = 0) const;

    // Exact fixed-recipe lookup. Returns 0 when the bag is not a known recipe.
    types::ItemID lookup_fixed(const std::vector<types::ConsumableID>& ingredient_ids) const;

    // Fixed recipe if there is one, otherwise the LCG approximation.
    types::ItemID craft_item(const std::vector<types::ConsumableID>& ingredient_ids);

    uint32_t get_run_seed() const { return run_seed_; }
    void set_seed(const std::string& run_seed) { run_seed_ = parse_seed(run_seed); }

    // The 32-bit start seed the crafting RNG actually runs on. Without it
    // craft_item() only answers for fixed recipes.
    void set_start_seed(uint32_t seed) { start_seeds_ = {seed}; }

    // A partly-narrowed search leaves several candidate seeds. That is still
    // worth having: for most bags every candidate rolls the SAME item, and then
    // the answer is certain regardless of which candidate is the real one.
    // craft_item() returns 0 when they disagree.
    void set_start_seeds(std::vector<uint32_t> seeds) { start_seeds_ = std::move(seeds); }
    bool have_start_seed() const { return !start_seeds_.empty(); }
    size_t start_seed_count() const { return start_seeds_.size(); }
    uint32_t start_seed() const { return start_seeds_.empty() ? 0 : start_seeds_.front(); }

    // Crafting result for a given start seed, without touching the object's own.
    types::ItemID craft_with_seed(const std::vector<types::ConsumableID>& sorted_ids,
                                  uint32_t start_seed) const;

    // The candidate distribution a bag produces. It depends only on the
    // ingredients, never on the seed, so the seed search builds it once per
    // observation and then only rolls -- which is what makes scanning all 2^32
    // seeds take seconds instead of hours.
    struct CraftDistribution {
        std::vector<types::ItemID> ids;    // ascending collectible ids
        std::vector<double> cumulative;    // running total of their weights
        double total = 0.0;
    };
    CraftDistribution build_distribution(const std::vector<types::ConsumableID>& sorted_ids) const;
    types::ItemID roll(const CraftDistribution& dist,
                       const std::vector<types::ConsumableID>& sorted_ids,
                       uint32_t start_seed) const;

    // Every start seed consistent with the observed crafts. Each observation is a
    // bag (8 component ids, any order) and the collectible it actually produced.
    // The first observation scans all 2^32 seeds; later ones filter the survivors.
    // `progress` is called with the fraction of the 2^32 scan done; `report` is
    // called after each observation with (observations used, seeds still alive),
    // which is what tells you whether another craft is worth recording.
    std::vector<uint32_t> find_start_seed(
        const std::vector<std::pair<std::vector<types::ConsumableID>, types::ItemID>>& observations,
        unsigned threads = 0,
        const std::function<void(double)>& progress = {},
        const std::function<void(size_t, size_t)>& report = {}) const;

    // The bag's total pickup value, per the game's own component table.
    int pickup_value_total(const std::vector<types::ConsumableID>& bag) const;

    const std::unordered_map<types::ConsumableID, ConsumableInfo>& consumables() const { return consumables_; }
    const std::unordered_map<types::ItemID, std::string>& item_names() const { return item_names_; }
    const std::unordered_map<types::ItemID, CollectibleInfo>& collectibles() const { return collectibles_; }
    const std::unordered_map<std::string, types::ItemID>& fixed_recipes() const { return fixed_crafts; }

    // The game's own 0-4 rating for a collectible; -1 when the id is not a
    // craftable collectible (blank, cut or unused ids in items.json).
    int item_quality(types::ItemID id) const;

    // Total pool points the bag contributes to each pool, per HASHING.md 4.
    std::unordered_map<std::string, int> pool_points(const std::vector<types::ConsumableID>& bag) const;

    // The pool the bag forces the roll into, or "" when none reaches the
    // threshold.
    std::string forced_pool(const std::vector<types::ConsumableID>& bag) const;

    // Collectibles whose real quality falls inside [lo, hi], restricted to
    // `pool` when non-empty. Sorted, so indexing it is deterministic. Cached:
    // enumerating a floor's candidate bags calls this thousands of times and
    // there are only a handful of distinct (band, pool) combinations.
    const std::vector<types::ItemID>& candidates(int lo, int hi, const std::string& pool) const;

private:
    uint32_t run_seed_;
    // Every start seed still consistent with what has been observed. One entry
    // means the run's seed is pinned; several mean the answer is only reported
    // when they agree.
    std::vector<uint32_t> start_seeds_;

    // Game data for the real algorithm, from resources/crafting_rng.json.
    std::vector<int> pickup_values_;                      // component id -> value
    std::vector<std::array<uint32_t, 3>> component_shifts_;  // component id -> xorshift triple

    // resources/itempools.json, in file order: crafting indexes pools by position.
    struct PoolEntry { types::ItemID id; double weight; };
    std::vector<std::vector<PoolEntry>> pools_;
    types::ItemID max_item_id_ = 0;
    // Scratch reused by craft_with_seed so the seed search does not allocate.
    mutable std::vector<double> item_weights_;
    std::unordered_map<std::string, types::ItemID> fixed_crafts;
    std::unordered_map<types::ConsumableID, ConsumableInfo> consumables_;
    std::unordered_map<types::ItemID, std::string> item_names_;
    std::unordered_map<types::ItemID, CollectibleInfo> collectibles_;
    // Sorted collectible ids that a pool can actually offer. Items belonging to
    // no pool (quest/unlockable) are excluded: crafting rolls out of a pool, so
    // an item no pool contains can never be the result.
    std::vector<types::ItemID> craftable_ids_;

    // Parse seed string to unsigned 32-bit integer using MD5.
    // Placeholder: the game does not derive its seed this way.
    uint32_t parse_seed(const std::string& seed_str) const;

    void load_crafting_tables(const std::string& rng_path, const std::string& pools_path);

    mutable std::unordered_map<std::string, std::vector<types::ItemID>> candidate_cache_;
};

#endif // BAG_OF_CRAFTING_HPP
