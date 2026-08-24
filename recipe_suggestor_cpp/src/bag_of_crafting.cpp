#include "bag_of_crafting.hpp"
#include <openssl/md5.h>
#include <atomic>
#include <fstream>
#include <thread>
#include <nlohmann/json.hpp>
#include <cstring>
#include <algorithm>
#include <sstream>
#include <iomanip>
#include "utils.hpp"
#include "constants.hpp"

BagOfCrafting::BagOfCrafting(const std::string& run_seed, const std::string& fixed_path,
                             const std::string& consumables_path, const std::string& items_path,
                             const std::string& collectibles_path)
    : run_seed_(parse_seed(run_seed)) {
    this->fixed_crafts = loadJsonToUnorderedMap(fixed_path);
    this->consumables_ = load_consumables(consumables_path);
    this->item_names_ = loadJsonToStringMap(items_path);
    this->collectibles_ = load_collectibles(collectibles_path);

    load_crafting_tables(constants::crafting_rng_path, constants::itempools_path);

    craftable_ids_.reserve(collectibles_.size());
    for (const auto& [id, info] : collectibles_) {
        if (!info.pools.empty()) craftable_ids_.push_back(id);
    }
    std::sort(craftable_ids_.begin(), craftable_ids_.end());
}

int BagOfCrafting::item_quality(types::ItemID id) const {
    auto it = collectibles_.find(id);
    return (it != collectibles_.end()) ? it->second.quality : -1;
}

std::unordered_map<std::string, int>
BagOfCrafting::pool_points(const std::vector<types::ConsumableID>& bag) const {
    std::unordered_map<std::string, int> points;
    for (types::ConsumableID id : bag) {
        auto it = consumables_.find(id);
        if (it == consumables_.end()) continue;
        const ConsumableInfo& info = it->second;
        if (!info.pool.empty() && info.pool_points > 0) {
            points[info.pool] += info.pool_points;
        }
    }
    return points;
}

std::string BagOfCrafting::forced_pool(const std::vector<types::ConsumableID>& bag) const {
    // HASHING.md 4: enough points in a pool force the roll into it. The
    // threshold is one full-strength contributor; ties break on the pool name so
    // the result does not depend on hash-map ordering.
    constexpr int threshold = 10;
    std::string best;
    int best_points = 0;
    for (const auto& [pool, pts] : pool_points(bag)) {
        if (pts < threshold) continue;
        if (pts > best_points || (pts == best_points && pool < best)) {
            best = pool;
            best_points = pts;
        }
    }
    return best;
}

const std::vector<types::ItemID>&
BagOfCrafting::candidates(int lo, int hi, const std::string& pool) const {
    const std::string key = std::to_string(lo) + ":" + std::to_string(hi) + ":" + pool;
    auto cached = candidate_cache_.find(key);
    if (cached != candidate_cache_.end()) return cached->second;

    std::vector<types::ItemID> out;
    for (types::ItemID id : craftable_ids_) {
        const CollectibleInfo& info = collectibles_.at(id);
        if (info.quality < lo || info.quality > hi) continue;
        if (!pool.empty() && !info.in_pool(pool)) continue;
        out.push_back(id);
    }
    return candidate_cache_.emplace(key, std::move(out)).first->second;
}

uint32_t BagOfCrafting::parse_seed(const std::string& seed_str) const {
    unsigned char digest[MD5_DIGEST_LENGTH];
    MD5(reinterpret_cast<const unsigned char*>(seed_str.c_str()),
        seed_str.length(),
        digest);

    uint32_t seed;
    std::memcpy(&seed, digest, sizeof(uint32_t));
    return seed;
}

int BagOfCrafting::calculate_quality_score(const std::vector<int>& qualities) const {
    int sum = 0;
    for (int quality : qualities) {
        sum += quality;
    }
    return sum;
}

int BagOfCrafting::bag_quality_score(const std::vector<types::ConsumableID>& bag) const {
    int sum = 0;
    for (types::ConsumableID id : bag) {
        auto it = consumables_.find(id);
        if (it != consumables_.end()) sum += it->second.quality;
    }
    return sum;
}

std::pair<int, int> BagOfCrafting::get_quality_range(int score, int pool_penalty) const {
    // The game's ladder. The Devil, Angel and Secret Room pools subtract 5
    // first, so the same bag can land in a different band per pool.
    const int n = score - pool_penalty;
    if (n > 34) return {4, 4};
    if (n > 26) return {3, 4};
    if (n > 22) return {2, 4};
    if (n > 18) return {2, 3};
    if (n > 14) return {1, 2};
    if (n > 8)  return {0, 2};
    return {0, 1};
}

types::ItemID BagOfCrafting::lookup_fixed(const std::vector<types::ConsumableID>& ingredient_ids) const {
    std::vector<types::ConsumableID> sorted_ids = ingredient_ids;
    // serializeBag sorts in place and is the one canonical key format; the
    // fixed.json keys are exactly this shape.
    std::string serial_bag = serializeBag(sorted_ids);

    auto it = fixed_crafts.find(serial_bag);
    return (it != fixed_crafts.end()) ? it->second : 0;
}

void BagOfCrafting::load_crafting_tables(const std::string& rng_path,
                                         const std::string& pools_path) {
    std::ifstream rng_file(rng_path);
    if (rng_file) {
        nlohmann::json j;
        rng_file >> j;
        for (const auto& v : j.at("pickup_values")) pickup_values_.push_back(v.get<int>());
        for (const auto& t : j.at("component_shifts")) {
            component_shifts_.push_back({t.at(0).get<uint32_t>(),
                                         t.at(1).get<uint32_t>(),
                                         t.at(2).get<uint32_t>()});
        }
    }

    std::ifstream pool_file(pools_path);
    if (pool_file) {
        nlohmann::json j;
        pool_file >> j;
        for (const auto& pool : j.at("pools")) {
            std::vector<PoolEntry> entries;
            for (const auto& item : pool.at("items")) {
                PoolEntry e{item.at(0).get<types::ItemID>(), item.at(1).get<double>()};
                entries.push_back(e);
                max_item_id_ = std::max(max_item_id_, e.id);
            }
            pools_.push_back(std::move(entries));
        }
    }
    item_weights_.assign(max_item_id_ + 1, 0.0);
}

int BagOfCrafting::pickup_value_total(const std::vector<types::ConsumableID>& bag) const {
    int total = 0;
    for (types::ConsumableID id : bag)
        if (id < pickup_values_.size()) total += pickup_values_[id];
    return total;
}

// The game's crafting RNG: a xorshift whose three shift amounts are swapped for
// every ingredient, so the ingredients themselves drive the state.
namespace {
inline uint32_t rng_next(uint32_t& state, const std::array<uint32_t, 3>& shift) {
    uint32_t num = state;
    num ^= num >> shift[0];
    num ^= num << shift[1];
    num ^= num >> shift[2];
    state = num;
    return num;
}
constexpr double kRngToFloat = 2.3283061589829401E-10;
}  // namespace

BagOfCrafting::CraftDistribution BagOfCrafting::build_distribution(
    const std::vector<types::ConsumableID>& sorted_ids) const {
    CraftDistribution dist;
    if (pools_.empty() || component_shifts_.empty()) return dist;

    int component_counts[64] = {0};
    int total_value = 0;
    for (types::ConsumableID id : sorted_ids) {
        if (id < 64) component_counts[id]++;
        if (id < pickup_values_.size()) total_value += pickup_values_[id];
    }

    // Which pools the bag can draw from, and how heavily. Treasure, Shop and
    // Boss are always in; the rest are unlocked by specific ingredients.
    struct PoolWeight { int idx; int weight; };
    const PoolWeight pool_weights[] = {
        {0, 1},                            // treasure
        {1, 2},                            // shop
        {2, 2},                            // boss
        {3, component_counts[3] * 10},     // devil       <- Black Hearts
        {4, component_counts[4] * 10},     // angel       <- Eternal Hearts
        {5, component_counts[6] * 5},      // secret      <- Bone Hearts
        {7, component_counts[29] * 10},    // shellGame   <- Poop Nuggets
        {8, component_counts[5] * 10},     // goldenChest <- Gold Hearts
        {9, component_counts[25] * 10},    // redChest    <- Cracked Keys
        {12, component_counts[7] * 10},    // curse       <- Rotten Hearts
        // Planetarium only counts when the bag holds no Red Heart, Penny, Key or Bomb.
        {26, (component_counts[1] + component_counts[8] + component_counts[12] +
              component_counts[15]) == 0 ? component_counts[23] * 10 : 0},
    };

    std::fill(item_weights_.begin(), item_weights_.end(), 0.0);
    for (const auto& pw : pool_weights) {
        if (pw.weight <= 0 || pw.idx >= (int)pools_.size()) continue;
        // Devil, Angel and Secret Room pools band 5 points lower.
        const int penalty = (pw.idx >= 3 && pw.idx <= 5) ? 5 : 0;
        const auto [quality_min, quality_max] = get_quality_range(total_value, penalty);

        for (const PoolEntry& entry : pools_[pw.idx]) {
            const int quality = item_quality(entry.id);
            if (quality < quality_min || quality > quality_max) continue;
            item_weights_[entry.id] += entry.weight * pw.weight;
        }
    }

    // Compact to the items that can actually come out, keeping ascending id
    // order: the game walks the weight table in that order.
    for (types::ItemID id = 1; id <= max_item_id_; ++id) {
        if (item_weights_[id] <= 0.0) continue;
        dist.total += item_weights_[id];
        dist.ids.push_back(id);
        dist.cumulative.push_back(dist.total);
    }
    return dist;
}

types::ItemID BagOfCrafting::roll(const CraftDistribution& dist,
                                  const std::vector<types::ConsumableID>& sorted_ids,
                                  uint32_t start_seed) const {
    if (dist.total <= 0.0) return 25;

    // Shift the RNG once per ingredient, in sorted order.
    uint32_t state = start_seed;
    std::array<uint32_t, 3> shift = component_shifts_[0];
    for (types::ConsumableID id : sorted_ids) {
        if (id < component_shifts_.size()) shift = component_shifts_[id];
        rng_next(state, shift);
    }
    // Every roll from here on uses one fixed shift triple.
    shift = component_shifts_[6];

    // Up to 20 attempts: the game retries when a roll lands on an item this run
    // cannot give (already taken, not unlocked). We cannot see the run's unlocks
    // or its emptied pools, so every pool item counts as available and the first
    // roll stands.
    for (int attempt = 0; attempt < 20; ++attempt) {
        const double target = rng_next(state, shift) * kRngToFloat * dist.total;
        auto it = std::upper_bound(dist.cumulative.begin(), dist.cumulative.end(), target);
        if (it != dist.cumulative.end())
            return dist.ids[std::distance(dist.cumulative.begin(), it)];
    }
    return 25;
}

types::ItemID BagOfCrafting::craft_with_seed(const std::vector<types::ConsumableID>& sorted_ids,
                                             uint32_t start_seed) const {
    return roll(build_distribution(sorted_ids), sorted_ids, start_seed);
}

types::ItemID BagOfCrafting::craft_item(const std::vector<types::ConsumableID>& ingredient_ids) {
    if (types::ItemID fixed = lookup_fixed(ingredient_ids)) return fixed;
    if (start_seeds_.empty()) return 0;

    std::vector<types::ConsumableID> sorted_ids = ingredient_ids;
    std::sort(sorted_ids.begin(), sorted_ids.end());

    // With the seed pinned this is one roll. With a candidate set it is one roll
    // per candidate, and only unanimity counts as an answer -- a bag whose
    // candidates disagree is exactly the bag we cannot call yet.
    const CraftDistribution dist = build_distribution(sorted_ids);
    const types::ItemID first = roll(dist, sorted_ids, start_seeds_.front());
    for (size_t i = 1; i < start_seeds_.size(); ++i)
        if (roll(dist, sorted_ids, start_seeds_[i]) != first) return 0;
    return first;
}

std::vector<uint32_t> BagOfCrafting::find_start_seed(
    const std::vector<std::pair<std::vector<types::ConsumableID>, types::ItemID>>& observations,
    unsigned threads,
    const std::function<void(double)>& progress,
    const std::function<void(size_t, size_t)>& report) const {

    std::vector<uint32_t> survivors;
    if (observations.empty()) return survivors;

    // Sorting once here keeps craft_with_seed allocation-free in the hot loop.
    std::vector<std::pair<std::vector<types::ConsumableID>, types::ItemID>> obs = observations;
    for (auto& o : obs) std::sort(o.first.begin(), o.first.end());

    if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency());

    // First observation: the only one that has to look at all 2^32 seeds.
    std::vector<std::vector<uint32_t>> per_thread(threads);
    std::atomic<uint64_t> done{0};
    const uint64_t span = (1ull << 32) / threads;

    // The distribution is seed-independent, so it is built once and every seed
    // only pays for the shifts and one lookup.
    const CraftDistribution dist = build_distribution(obs[0].first);
    if (dist.total <= 0.0) return survivors;

    auto scan = [&](unsigned t) {
        const uint64_t begin = span * t;
        const uint64_t end = (t + 1 == threads) ? (1ull << 32) : span * (t + 1);
        for (uint64_t s = begin; s < end; ++s) {
            if (roll(dist, obs[0].first, static_cast<uint32_t>(s)) == obs[0].second)
                per_thread[t].push_back(static_cast<uint32_t>(s));
            if (progress && (s & 0xFFFFFF) == 0) {
                const uint64_t d = done.fetch_add(0x1000000) + 0x1000000;
                progress(static_cast<double>(d) / static_cast<double>(1ull << 32));
            }
        }
    };

    std::vector<std::thread> pool;
    for (unsigned t = 0; t < threads; ++t) pool.emplace_back(scan, t);
    for (auto& th : pool) th.join();
    size_t total_matches = 0;
    for (auto& v : per_thread) total_matches += v.size();
    survivors.reserve(total_matches);
    for (auto& v : per_thread) {
        survivors.insert(survivors.end(), v.begin(), v.end());
        v.clear();
        v.shrink_to_fit();
    }
    std::sort(survivors.begin(), survivors.end());
    if (report) report(1, survivors.size());

    // Every later observation just filters what is left.
    for (size_t i = 1; i < obs.size() && survivors.size() > 1; ++i) {
        std::vector<uint32_t> kept;
        for (uint32_t s : survivors)
            if (craft_with_seed(obs[i].first, s) == obs[i].second) kept.push_back(s);
        survivors.swap(kept);
    }
    return survivors;
}

