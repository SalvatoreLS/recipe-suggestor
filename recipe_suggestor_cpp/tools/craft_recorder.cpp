// Record crafting observations from the live game, so the bags that go into
// find_start_seed are read off the screen instead of typed from memory.
//
//   ./craft_recorder [--seconds N] [--out DIR]
//
// It watches the bag, prints every change, and when a full bag of 8 empties --
// which is what crafting looks like -- it writes the bag that was there just
// before, plus the frame captured right after, so the collectible that popped
// out can be identified. Each recorded craft becomes one line of an
// observations file, needing only the item id filled in.
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include "constants.hpp"
#include "pipeline/nodes/boc_detector.hpp"
#include "pipeline/nodes/frame_capturer.hpp"
#include "pipeline/nodes/recipe_suggestor.hpp"
#include "pipeline/nodes/router.hpp"
#include "utils.hpp"

namespace {
std::atomic<bool> running{true};
void on_sigint(int) { running = false; }
}  // namespace

int main(int argc, char** argv) {
    int seconds = 900;
    std::string out_dir = "crafts";
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seconds" && i + 1 < argc) seconds = std::stoi(argv[++i]);
        else if (arg == "--out" && i + 1 < argc) out_dir = argv[++i];
    }
    std::filesystem::create_directories(out_dir);
    std::signal(SIGINT, on_sigint);

    BoCDetector detector(constants::boc_model_path, constants::img_width, constants::img_height);
    RecipeSuggestor suggestor;
    suggestor.bind_models(detector.class_names(), {});
    Router router;
    FrameCapturer capturer;

    std::ofstream obs(out_dir + "/observations.txt", std::ios::app);
    obs << "# item id (fill in), then the 8 components read off the screen\n";

    std::cout << "Recording for " << seconds << "s. Play; craft whenever you like.\n"
              << "Ctrl-C to stop early.\n\n";

    std::vector<types::ConsumableID> last_bag;
    int crafts = 0;
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(seconds);

    while (running && std::chrono::steady_clock::now() < deadline) {
        ScreenCapture* screen = capturer.capture_screen();
        if (!screen) break;

        ScreenCapture floor_img = {nullptr, 0, 0};
        ScreenCapture boc_img = {nullptr, 0, 0};
        router.route(screen, &floor_img, &boc_img);

        std::vector<Prediction> preds;
        auto detected = detector.detect(screen_capture_to_mat(boc_img), &preds);
        auto bag = suggestor.translate_bag(detected.snapshot());

        if (bag != last_bag) {
            // A full bag emptying is a craft. Anything else is just picking up.
            const bool crafted = last_bag.size() == constants::bag_size && bag.empty();
            std::cout << "bag " << bag.size() << "/8: ";
            for (size_t i = 0; i < bag.size(); ++i)
                std::cout << (i ? ", " : "") << suggestor.consumable_name(bag[i]);
            if (bag.empty()) std::cout << "(empty)";
            std::cout << "\n";

            if (crafted) {
                crafts++;
                const std::string stem = out_dir + "/craft_" + std::to_string(crafts);
                cv::imwrite(stem + "_after.png", screen_capture_to_mat(*screen));

                obs << "???? ";
                for (size_t i = 0; i < last_bag.size(); ++i) obs << (i ? " " : "") << last_bag[i];
                obs << "   # craft " << crafts << ": ";
                for (size_t i = 0; i < last_bag.size(); ++i)
                    obs << (i ? ", " : "") << suggestor.consumable_name(last_bag[i]);
                obs << "\n" << std::flush;

                std::cout << "  ^ CRAFT " << crafts << " recorded (bag written to "
                          << out_dir << "/observations.txt, frame to " << stem << "_after.png)\n";
            }
            last_bag = bag;
        }

        delete screen;
        std::this_thread::sleep_for(std::chrono::milliseconds(150));
    }

    std::cout << "\nDone: " << crafts << " craft(s) recorded in " << out_dir
              << "/observations.txt\nFill in each item id, then run find_start_seed on it.\n";
    return 0;
}
