#include "pipeline/nodes/recipe_suggestor.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <unistd.h>
#include <iostream>
#include <set>
#include <sstream>

RecipeSuggestor::RecipeSuggestor(const std::string& run_seed,
                                 const std::string& class_map_path,
                                 const std::string& fixed_path,
                                 const std::string& consumables_path,
                                 const std::string& items_path)
    : crafter_(run_seed, fixed_path, consumables_path, items_path) {
    // Empty model names skip the tripwire; bind_models() arms it once the
    // detectors are up.
    boc_map_ = load_class_map(class_map_path, "boc", {});
    floor_map_ = load_class_map(class_map_path, "floor", {});
}

void RecipeSuggestor::bind_models(const std::vector<std::string>& boc_names,
                                  const std::vector<std::string>& floor_names) {
    if (!boc_names.empty()) {
        boc_map_ = load_class_map(constants::class_map_path, "boc", boc_names);
    }
    if (!floor_names.empty()) {
        floor_map_ = load_class_map(constants::class_map_path, "floor", floor_names);
    }
}

std::string RecipeSuggestor::consumable_name(types::ConsumableID id) const {
    const auto& table = crafter_.consumables();
    auto it = table.find(id);
    return (it != table.end()) ? it->second.name : ("consumable#" + std::to_string(id));
}

std::vector<types::ConsumableID> RecipeSuggestor::translate_bag(
    const std::vector<types::ConsumableID>& boc_classes) const {
    std::vector<types::ConsumableID> bag;
    for (types::ConsumableID cls : boc_classes) {
        if (cls >= boc_map_.size()) continue;
        for (const auto& [cid, qty] : boc_map_[cls].consumables) {
            for (types::Quantity n = 0; n < qty; ++n) bag.push_back(cid);
        }
    }
    // A spurious extra detection must not poison the bag.
    if (bag.size() > constants::bag_size) bag.resize(constants::bag_size);
    return bag;
}

std::map<types::ConsumableID, types::Quantity> RecipeSuggestor::translate_floor(
    const std::map<types::ConsumableID, types::Quantity>& floor_classes) const {
    std::map<types::ConsumableID, types::Quantity> supply;
    for (const auto& [cls, count] : floor_classes) {
        if (cls >= floor_map_.size()) continue;
        for (const auto& [cid, qty] : floor_map_[cls].consumables) {
            int total = supply[cid] + static_cast<int>(qty) * static_cast<int>(count);
            supply[cid] = static_cast<types::Quantity>(std::min(total, 255));
        }
    }
    return supply;
}

// The floor as a list of things you can walk over, each with what it is worth
// in the bag. Going through the class map here (rather than through
// translate_floor's flattened unit counts) is what keeps a pickup indivisible:
// a Soul Heart is two units, and a plan must take both or neither.
std::vector<RecipeSuggestor::FloorPickup> RecipeSuggestor::floor_pickups(
    const std::map<types::ConsumableID, types::Quantity>& floor_classes) const {
    std::vector<FloorPickup> out;
    for (const auto& [cls, count] : floor_classes) {
        if (cls >= floor_map_.size() || count == 0) continue;
        FloorPickup p;
        // Class-map labels are snake_case ("soul_heart"); the console shows
        // them to a player mid-run, so title them.
        p.name = floor_map_[cls].name;
        bool start = true;
        for (char& c : p.name) {
            if (c == '_') { c = ' '; start = true; continue; }
            if (start) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
            start = false;
        }
        for (const auto& [cid, qty] : floor_map_[cls].consumables) {
            for (types::Quantity n = 0; n < qty; ++n) p.units.push_back(cid);
        }
        if (p.units.empty()) continue;  // 'extra' and friends carry nothing
        p.available = count;
        out.push_back(std::move(p));
    }
    // Cheapest pickups first, so the search prunes sooner.
    std::sort(out.begin(), out.end(), [](const FloorPickup& a, const FloorPickup& b) {
        return a.units.size() < b.units.size();
    });
    return out;
}

// Depth-first over pickup types, taking 0..available of each. A combination is
// a plan when the units it adds leave the bag exactly full: the bag holds 8 and
// takes from the front, so units beyond the free slots push out that many of
// the oldest entries. That overflow is the whole reason a full bag is still
// worth planning for.
void RecipeSuggestor::enumerate_plans(const std::vector<types::ConsumableID>& bag,
                                      const std::vector<FloorPickup>& floor,
                                      size_t index,
                                      std::vector<std::string>& chosen_names,
                                      std::vector<types::ConsumableID>& chosen_units,
                                      std::vector<Suggestion>& out) const {
    if (out.size() >= constants::max_suggestion_candidates) return;

    const size_t added = chosen_units.size();
    // Picking up more than a bagful just discards earlier pickups: never useful.
    if (added > constants::bag_size) return;

    if (bag.size() + added >= constants::bag_size) {
        const size_t overflow = bag.size() + added - constants::bag_size;
        Suggestion s;
        s.to_drop.assign(bag.begin(), bag.begin() + overflow);
        s.to_add = chosen_units;
        s.pickups = chosen_names;
        s.recipe.assign(bag.begin() + overflow, bag.end());
        s.recipe.insert(s.recipe.end(), chosen_units.begin(), chosen_units.end());
        if (s.recipe.size() == constants::bag_size) {
            std::sort(s.recipe.begin(), s.recipe.end());
            std::sort(s.to_add.begin(), s.to_add.end());
            std::sort(s.pickups.begin(), s.pickups.end());
            out.push_back(std::move(s));
        }
    }

    if (index >= floor.size()) return;

    // Skip this pickup type entirely, then take one more of it each round.
    enumerate_plans(bag, floor, index + 1, chosen_names, chosen_units, out);
    const FloorPickup& p = floor[index];
    for (types::Quantity taken = 1; taken <= p.available; ++taken) {
        chosen_names.push_back(p.name);
        chosen_units.insert(chosen_units.end(), p.units.begin(), p.units.end());
        if (chosen_units.size() > constants::bag_size) break;
        enumerate_plans(bag, floor, index + 1, chosen_names, chosen_units, out);
    }
    // Unwind whatever this level pushed.
    while (chosen_names.size() > 0 && chosen_names.back() == p.name &&
           chosen_units.size() >= p.units.size()) {
        bool ours = true;
        for (size_t k = 0; k < p.units.size(); ++k)
            if (chosen_units[chosen_units.size() - p.units.size() + k] != p.units[k]) ours = false;
        if (!ours) break;
        chosen_names.pop_back();
        chosen_units.resize(chosen_units.size() - p.units.size());
    }
}

std::vector<Suggestion> RecipeSuggestor::suggest(
    const std::vector<types::ConsumableID>& boc_classes,
    const std::map<types::ConsumableID, types::Quantity>& floor_classes) {

    std::vector<types::ConsumableID> bag = translate_bag(boc_classes);
    const std::vector<FloorPickup> floor = floor_pickups(floor_classes);

    // Memo key: the bag plus the floor. The bag keeps its ORDER here -- with a
    // full bag, which items get pushed out depends on it, so two bags with the
    // same contents in a different order are different inputs.
    std::ostringstream sig;
    for (types::ConsumableID id : bag) sig << id << ',';
    sig << '|';
    for (const auto& p : floor) sig << p.name << 'x' << static_cast<int>(p.available) << ';';
    // Knowing the seed changes the output for identical inputs, so it belongs
    // in the key.
    sig << '|' << (crafter_.have_start_seed() ? 's' : '-');
    if (sig.str() == last_signature_) return last_result_;
    last_signature_ = sig.str();
    computations_++;

    last_floor_names_.clear();
    for (const auto& p : floor)
        for (types::Quantity n = 0; n < p.available; ++n) last_floor_names_.push_back(p.name);
    std::sort(last_floor_names_.begin(), last_floor_names_.end());

    std::vector<Suggestion> results;
    if (bag.empty() && floor.empty()) {
        last_result_ = results;
        return results;
    }

    std::vector<Suggestion> plans;
    std::vector<std::string> names;
    std::vector<types::ConsumableID> units;
    enumerate_plans(bag, floor, 0, names, units, plans);
    if (plans.size() >= constants::max_suggestion_candidates) {
        std::cerr << "[RecipeSuggestor] candidate cap hit; results are truncated.\n";
    }

    // Different routes can end at the same bag (replacing a penny with a penny).
    // Keep the one that asks for the fewest pickups.
    std::map<std::string, size_t> best_by_bag;
    for (size_t i = 0; i < plans.size(); ++i) {
        std::vector<types::ConsumableID> key_bag = plans[i].recipe;
        const std::string key = serializeBag(key_bag);
        auto it = best_by_bag.find(key);
        if (it == best_by_bag.end() || plans[i].pickups.size() < plans[it->second].pickups.size()) {
            best_by_bag[key] = i;
        }
    }

    for (const auto& [key, idx] : best_by_bag) {
        (void)key;
        Suggestion s = plans[idx];
        s.quality_score = crafter_.bag_quality_score(s.recipe);
        const auto [lo, hi] = crafter_.get_quality_range(s.quality_score);
        s.band_lo = lo;
        s.band_hi = hi;
        s.pool = crafter_.forced_pool(s.recipe);

        if (types::ItemID fixed = crafter_.lookup_fixed(s.recipe)) {
            s.item_id = fixed;
            s.exact = true;
        } else {
            // craft_item returns 0 when the run's start seed is unknown. The
            // plan and its quality band are still real, so report it with no
            // item rather than dropping it: which pickups raise the tier is the
            // actionable part either way.
            s.item_id = crafter_.craft_item(s.recipe);
            s.exact = false;
        }

        if (s.item_id) {
            const auto& item_names = crafter_.item_names();
            auto it = item_names.find(s.item_id);
            s.item_name = (it != item_names.end()) ? it->second
                                                   : ("Item #" + std::to_string(s.item_id));
        }
        results.push_back(std::move(s));
    }

    std::sort(results.begin(), results.end(), [](const Suggestion& a, const Suggestion& b) {
        if (a.exact != b.exact) return a.exact;                       // guaranteed first
        if (a.quality_score != b.quality_score) return a.quality_score > b.quality_score;
        if (a.pickups.size() != b.pickups.size()) return a.pickups.size() < b.pickups.size();
        if (a.recipe != b.recipe) return a.recipe < b.recipe;
        return a.item_id < b.item_id;                                 // deterministic tie-break
    });

    if (results.size() > constants::max_elements_rank) results.resize(constants::max_elements_rank);

    last_result_ = results;
    return results;
}

// Console rendering. The output is read mid-run, out of the corner of an eye,
// so it answers one question first -- what do I walk over next -- and keeps the
// supporting detail underneath it.
namespace {

// ANSI styling, dropped when stdout is not a terminal (piping to a file, the
// test harness) or when NO_COLOR is set.
struct Style {
    const char* dim = "";
    const char* bold = "";
    const char* action = "";
    const char* good = "";
    const char* reset = "";
};

const Style& style() {
    static const Style s = [] {
        Style st;
        if (isatty(fileno(stdout)) && !std::getenv("NO_COLOR")) {
            st.dim = "\x1b[2m";
            st.bold = "\x1b[1m";
            st.action = "\x1b[1;32m";   // bright green: the thing to do
            st.good = "\x1b[33m";       // yellow: the quality it buys
            st.reset = "\x1b[0m";
        }
        return st;
    }();
    return s;
}

std::string rule() { return std::string(64, '-'); }

}  // namespace

std::string RecipeSuggestor::format_state(
        const std::vector<types::ConsumableID>& boc_classes,
        const std::map<types::ConsumableID, types::Quantity>& floor_classes) const {
    const Style& st = style();
    const auto bag = translate_bag(boc_classes);
    const auto floor_units = translate_floor(floor_classes);

    // Name the pickups, not the units they expand to.
    std::vector<std::string> names;
    for (const auto& p : floor_pickups(floor_classes))
        for (types::Quantity n = 0; n < p.available; ++n) names.push_back(p.name);
    std::sort(names.begin(), names.end());

    std::ostringstream out;
    out << format_header(bag, join_names(names, ", "));

    size_t supply = 0;
    for (const auto& [id, qty] : floor_units) supply += qty;
    (void)supply;
    const size_t missing = bag.size() >= constants::bag_size ? 0 : constants::bag_size - bag.size();

    out << "      " << st.dim;
    if (missing > 0) {
        out << "nothing craftable yet: the bag needs " << missing << " more";
        if (names.empty()) out << ", and the floor is empty";
        else out << ", and no combination of what is on the floor fills it";
    } else {
        out << "nothing craftable from this bag";
    }
    out << st.reset << "\n";
    return out.str();
}

std::string RecipeSuggestor::join_counted(const std::vector<types::ConsumableID>& ids,
                                          const std::string& sep) const {
    std::vector<std::pair<types::ConsumableID, int>> runs;
    for (types::ConsumableID id : ids) {
        if (!runs.empty() && runs.back().first == id) runs.back().second++;
        else runs.push_back({id, 1});
    }
    std::string out;
    for (size_t i = 0; i < runs.size(); ++i) {
        if (i) out += sep;
        out += consumable_name(runs[i].first);
        if (runs[i].second > 1) out += " x" + std::to_string(runs[i].second);
    }
    return out;
}

// The two-line "here is the situation" block both outputs open with.
std::string RecipeSuggestor::join_names(const std::vector<std::string>& names,
                                        const std::string& sep) {
    std::vector<std::pair<std::string, int>> runs;
    for (const auto& n : names) {
        if (!runs.empty() && runs.back().first == n) runs.back().second++;
        else runs.push_back({n, 1});
    }
    std::string out;
    for (size_t i = 0; i < runs.size(); ++i) {
        if (i) out += sep;
        out += runs[i].first;
        if (runs[i].second > 1) out += " x" + std::to_string(runs[i].second);
    }
    return out;
}

std::string RecipeSuggestor::format_header(const std::vector<types::ConsumableID>& bag,
                                           const std::string& floor_line) const {
    const Style& st = style();
    std::ostringstream out;
    out << st.dim << rule() << st.reset << "\n";
    out << st.bold << "BAG   " << bag.size() << "/" << constants::bag_size << st.reset
        << st.dim << "   quality " << crafter_.bag_quality_score(bag) << st.reset << "\n";
    // Slot order matters -- the leftmost entry is the oldest and the first to
    // be pushed out -- so this lists slots, it does not collapse duplicates.
    out << "      ";
    if (bag.empty()) {
        out << "(empty)";
    } else {
        for (size_t i = 0; i < bag.size(); ++i) {
            if (i) out << "  ";
            out << consumable_name(bag[i]);
        }
        out << st.dim << "   (oldest first)" << st.reset;
    }
    out << "\n";
    out << st.bold << "FLOOR" << st.reset << " " << (floor_line.empty() ? "(nothing)" : floor_line) << "\n";
    return out.str();
}

std::string RecipeSuggestor::format(const std::vector<Suggestion>& suggestions) const {
    if (suggestions.empty()) return {};

    const Style& st = style();
    const auto& best = suggestions.front();

    // Reconstruct the current bag from the top plan: what its recipe holds,
    // minus what it still has to collect, plus what it would push out.
    std::vector<types::ConsumableID> bag = best.recipe;
    for (types::ConsumableID id : best.to_add) {
        auto it = std::find(bag.begin(), bag.end(), id);
        if (it != bag.end()) bag.erase(it);
    }
    bag.insert(bag.end(), best.to_drop.begin(), best.to_drop.end());
    const int current_q = crafter_.bag_quality_score(bag);

    std::ostringstream out;
    out << format_header(bag, join_names(last_floor_names_, ", ")) << "\n";

    // The headline: one instruction, and what it buys.
    out << "  " << st.action << ">> ";
    if (best.pickups.empty()) out << "CRAFT NOW";
    else out << "PICK UP  " << join_names(best.pickups);
    out << st.reset;

    out << "   " << st.good << "quality " << current_q << " -> " << best.quality_score << st.reset
        << st.dim << "  (tier " << best.band_lo;
    if (best.band_hi != best.band_lo) out << "-" << best.band_hi;
    if (!best.pool.empty()) out << ", " << best.pool << " pool";
    out << ")" << st.reset << "\n";

    if (!best.to_drop.empty())
        out << "     " << st.dim << "pushes out the oldest: " << join_counted(best.to_drop, ", ")
            << st.reset << "\n";
    if (best.item_id)
        out << "     " << (best.exact ? "* " : "  ") << "crafts: " << best.item_name << "\n";

    // Everything else, compressed to one line each.
    const size_t shown = std::min(suggestions.size(), constants::max_shown_alternatives + 1);
    if (shown > 1) {
        out << "\n     " << st.dim << "or" << st.reset << "\n";
        for (size_t i = 1; i < shown; ++i) {
            const auto& s = suggestions[i];
            std::ostringstream line;
            line << (s.pickups.empty() ? std::string("craft now") : join_names(s.pickups));
            std::string what = line.str();
            if (what.size() < 34) what.resize(34, ' ');
            out << "      " << (s.exact ? "*" : " ") << " " << what
                << st.dim << "q " << s.quality_score << "  tier " << s.band_lo;
            if (s.band_hi != s.band_lo) out << "-" << s.band_hi;
            if (!s.to_drop.empty()) out << "   drops " << join_counted(s.to_drop, ", ");
            out << st.reset << "\n";
        }
    }

    if (!crafter_.have_start_seed())
        out << "\n     " << st.dim
            << "item names need the run's crafting seed: see find_start_seed, then --start-seed."
            << st.reset << "\n";

    return out.str();
}
