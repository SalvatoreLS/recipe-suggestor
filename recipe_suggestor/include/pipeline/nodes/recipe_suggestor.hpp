#ifndef RECIPE_SUGGESTOR_HPP
#define RECIPE_SUGGESTOR_HPP

#include <map>
#include <string>
#include <sys/types.h>
#include <unordered_map>
#include <vector>

#include "bag_of_crafting.hpp"
#include "constants.hpp"
#include "data_structures/circular_list.hpp"
#include "types.hpp"
#include "utils.hpp"

struct Suggestion {
    // 0 when the item is not knowable: no fixed recipe matched and the run's
    // start seed is unknown. The plan is still worth showing -- the quality
    // band and the pool below are real game data even when the item is not.
    types::ItemID item_id = 0;
    std::string item_name;
    std::vector<types::ConsumableID> recipe;  // the full 8, sorted ascending
    std::vector<types::ConsumableID> to_add;  // the bag units those pickups add
    // What to physically walk over, by floor-pickup name. One pickup can be
    // worth several bag units -- a Soul Heart is two, a double heart four -- so
    // this is the actionable list and `to_add` is its expansion.
    std::vector<std::string> pickups;
    // Items pushed out of the bag by those pickups: the bag holds 8 and takes
    // from the front, so picking up past full costs the oldest entries.
    std::vector<types::ConsumableID> to_drop;
    int quality_score = 0;
    // The real quality band this bag rolls in, and the pool it is forced into
    // ("" when nothing reaches the threshold). Both come from the game's own
    // data, so they hold whether or not the item id does.
    int band_lo = 0;
    int band_hi = 0;
    std::string pool;
    // true  -> came from resources/fixed.json, correct in every run
    // false -> computed from the run seed (or unknown, when item_id is 0)
    bool exact = false;
};

class RecipeSuggestor {
public:
    explicit RecipeSuggestor(const std::string& run_seed = "",
                             const std::string& class_map_path = constants::class_map_path,
                             const std::string& fixed_path = constants::fixed_crafts_path,
                             const std::string& consumables_path = constants::consumables_path,
                             const std::string& items_path = constants::item_names_path);

    // Cross-checks the class maps against what the models actually report.
    // Throws when they disagree, which means the wrong .onnx file is loaded.
    void bind_models(const std::vector<std::string>& boc_names,
                     const std::vector<std::string>& floor_names);

    void set_seed(const std::string& run_seed) { crafter_.set_seed(run_seed); }

    // The run's 32-bit crafting seed. Item names can only be computed with it;
    // without it every plan still carries its real quality, tier and pool.
    void set_start_seed(uint32_t seed) { crafter_.set_start_seed(seed); }
    // A partly-narrowed candidate set still names the items its members agree on.
    void set_start_seeds(std::vector<uint32_t> seeds) { crafter_.set_start_seeds(std::move(seeds)); }
    bool have_start_seed() const { return crafter_.have_start_seed(); }
    size_t start_seed_count() const { return crafter_.start_seed_count(); }

    // Inputs are detector CLASS INDICES, not consumable ids; translation
    // happens here. Takes plain containers so the caller can snapshot under its
    // lock and run the actual work outside it.
    std::vector<Suggestion> suggest(const std::vector<types::ConsumableID>& boc_classes,
                                    const std::map<types::ConsumableID, types::Quantity>& floor_classes);

    std::string format(const std::vector<Suggestion>& suggestions) const;

    // What the detectors currently see, rendered for the console. suggest()
    // legitimately returns nothing most of the time -- an 8-slot recipe needs
    // more pickups than a room usually holds -- and printing nothing at all
    // makes a working pipeline look like a hung one. Takes the same raw class
    // indices as suggest().
    std::string format_state(const std::vector<types::ConsumableID>& boc_classes,
                             const std::map<types::ConsumableID, types::Quantity>& floor_classes) const;

    // Exposed for testing and for the console output.
    std::vector<types::ConsumableID> translate_bag(const std::vector<types::ConsumableID>& boc_classes) const;
    std::map<types::ConsumableID, types::Quantity> translate_floor(
        const std::map<types::ConsumableID, types::Quantity>& floor_classes) const;
    std::string consumable_name(types::ConsumableID id) const;
    // Consumable list with runs collapsed: "Soul Heart x2 + Penny".
    std::string join_counted(const std::vector<types::ConsumableID>& ids,
                             const std::string& sep) const;
    // The same for pickup names: "Soul Heart + Penny x2".
    static std::string join_names(const std::vector<std::string>& names,
                                  const std::string& sep = " + ");
    // The "here is the situation" block both outputs open with.
    std::string format_header(const std::vector<types::ConsumableID>& bag,
                              const std::string& floor_line) const;

    size_t computations() const { return computations_; } // memo-cache probe

private:
    std::vector<ClassEntry> boc_map_;
    std::vector<ClassEntry> floor_map_;
    BagOfCrafting crafter_;

    // One-entry memo. The inputs only change when the game state does, so at
    // pipeline speed this short-circuits nearly every call.
    std::string last_signature_;
    // The floor as suggest() last saw it, so format() can show what is actually
    // lying there (with counts) rather than inferring it from the plans.
    std::vector<std::string> last_floor_names_;
    std::vector<Suggestion> last_result_;
    size_t computations_ = 0;

    // One thing lying on the floor, and what it is worth in the bag. Pickups
    // are indivisible: a plan may take a whole Soul Heart (2 units) or none of
    // it, never one unit of it.
    struct FloorPickup {
        std::string name;
        std::vector<types::ConsumableID> units;
        types::Quantity available = 0;
    };

    std::vector<FloorPickup> floor_pickups(
        const std::map<types::ConsumableID, types::Quantity>& floor_classes) const;

    // Every combination of whole pickups that leaves the bag full, as plans.
    void enumerate_plans(const std::vector<types::ConsumableID>& bag,
                         const std::vector<FloorPickup>& floor,
                         size_t index,
                         std::vector<std::string>& chosen_names,
                         std::vector<types::ConsumableID>& chosen_units,
                         std::vector<Suggestion>& out) const;
};

#endif // RECIPE_SUGGESTOR_HPP
