// Recover the run's 32-bit start seed from crafts you have actually made.
//
// The crafting RNG runs on the run's start seed, a number the game never shows;
// the seed string on the pause screen is an encoding of it that is not public.
// So instead of decoding the string, this searches for the seed that reproduces
// what the game actually gave you.
//
//   ./find_start_seed observations.txt [--threads N]
//
// Each line of the file is one craft: the collectible id it produced, then the
// 8 component ids that were in the bag, in any order.
//
//   # item  components
//   177     8 8 8 8 8 8 8 8
//
// One craft leaves millions of candidate seeds; each further craft cuts that by
// roughly the number of items the bag could have produced, so three or four
// crafts usually pin it exactly.
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "bag_of_crafting.hpp"
#include "constants.hpp"

int main(int argc, char** argv) {
    std::string path;
    std::string dump_path;
    unsigned threads = 0;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--threads" && i + 1 < argc) threads = std::stoul(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) dump_path = argv[++i];
        else path = arg;
    }
    if (path.empty()) {
        std::cerr << "Usage: " << argv[0] << " <observations.txt> [--threads N]\n";
        return 2;
    }

    std::ifstream in(path);
    if (!in) {
        std::cerr << "Cannot read " << path << "\n";
        return 1;
    }

    std::vector<std::pair<std::vector<types::ConsumableID>, types::ItemID>> observations;
    std::string line;
    while (std::getline(in, line)) {
        if (auto hash = line.find('#'); hash != std::string::npos) line.erase(hash);
        std::istringstream is(line);
        int item;
        if (!(is >> item)) continue;
        std::vector<types::ConsumableID> comps;
        int c;
        while (is >> c) comps.push_back(static_cast<types::ConsumableID>(c));
        if (comps.size() != constants::bag_size) {
            std::cerr << "Skipping a line with " << comps.size() << " components (need "
                      << constants::bag_size << ")\n";
            continue;
        }
        observations.push_back({comps, static_cast<types::ItemID>(item)});
    }
    if (observations.empty()) {
        std::cerr << "No usable observations in " << path << "\n";
        return 1;
    }
    std::cout << "Searching 2^32 seeds against " << observations.size() << " craft(s)...\n";

    BagOfCrafting boc("unused");
    int last_percent = -1;
    auto seeds = boc.find_start_seed(observations, threads,
      [&](double fraction) {
        const int percent = static_cast<int>(fraction * 100);
        if (percent != last_percent && percent % 5 == 0) {
            last_percent = percent;
            std::cerr << "\r  " << percent << "%   " << std::flush;
        }
      },
      [](size_t used, size_t alive) {
          std::cerr << "\r          \r";
          std::cout << "  after craft " << used << ": " << alive << " seed(s) still possible\n"
                    << std::flush;
      });
    std::cerr << "\r          \r";

    std::cout << seeds.size() << " seed(s) match every craft.\n";
    if (!dump_path.empty()) {
        std::ofstream out(dump_path);
        for (uint32_t s : seeds) out << s << "\n";
        std::cout << "Wrote every candidate to " << dump_path << "\n";
    }
    if (seeds.empty()) {
        std::cout << "\nNothing matched. Either a component id or an item id is wrong, or the\n"
                     "craft used an item this run had already taken (the game rerolls those,\n"
                     "and we cannot see which ones they are).\n";
        return 1;
    }
    for (size_t i = 0; i < seeds.size() && i < 10; ++i)
        std::cout << "  " << seeds[i] << "\n";
    if (seeds.size() > 10) std::cout << "  ... and " << (seeds.size() - 10) << " more\n";

    if (seeds.size() == 1) {
        std::cout << "\nRun with:  --start-seed " << seeds.front() << "\n";
    } else {
        std::cout << "\nToo many still match. Craft once more and add the result as another line.\n";
    }
    return 0;
}
