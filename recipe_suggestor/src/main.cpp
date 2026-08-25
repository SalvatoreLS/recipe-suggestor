#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include "pipeline/pipeline.hpp"
#include "pipeline/nodes/boc_detector.hpp"
#include "pipeline/nodes/floor_detector.hpp"
#include "pipeline/nodes/recipe_suggestor.hpp"
#include "pipeline/nodes/router.hpp"

namespace {

std::atomic<bool> g_interrupted{false};

void handle_sigint(int) {
    // Only async-signal-safe work here: flip a flag and let main() do the rest.
    g_interrupted = true;
}

// --start-seed takes either the number itself or the file find_start_seed wrote,
// so a half-narrowed search is usable without editing anything by hand.
std::vector<uint32_t> parse_start_seeds(const std::string& arg) {
    std::vector<uint32_t> seeds;
    if (!arg.empty() && std::all_of(arg.begin(), arg.end(), ::isdigit)) {
        seeds.push_back(static_cast<uint32_t>(std::stoul(arg)));
        return seeds;
    }
    std::ifstream in(arg);
    std::string line;
    while (std::getline(in, line)) {
        if (auto hash = line.find('#'); hash != std::string::npos) line.erase(hash);
        try {
            size_t pos = 0;
            unsigned long v = std::stoul(line, &pos);
            if (pos > 0) seeds.push_back(static_cast<uint32_t>(v));
        } catch (const std::exception&) {}
    }
    return seeds;
}

void usage(const char* argv0) {
    std::cout <<
        "Usage: " << argv0 << " [options]\n"
        "  --seed \"<run seed>\"   Isaac run seed, e.g. \"7W2N L9AK\" (needed for crafting)\n"
        "  --replay <dir>        run over images in <dir> instead of capturing the screen\n"
        "  --start-seed <n|file> the run's 32-bit crafting seed, needed to name the crafted\n"
        "                        item; recover it with tools/find_start_seed (see README).\n"
        "                        Also takes that tool's --out file: while several seeds are\n"
        "                        still possible, items they agree on are still named\n"
        "                        without it only exact fixed.json recipes are shown)\n"
        "  --help                this message\n";
}

// Offline mode: no X11, no running game, reproducible. A full-screen-looking
// image goes through the Router first; anything smaller is treated as an
// already-cropped bag image (which is what the dataset test images are).
int run_replay(const std::string& dir, const std::string& seed, const std::vector<uint32_t>& start_seeds) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(dir)) {
        std::cerr << "Not a directory: " << dir << "\n";
        return 1;
    }

    std::unique_ptr<BoCDetector> boc;
    try {
        boc = std::make_unique<BoCDetector>(constants::boc_model_path,
                                            constants::img_width, constants::img_height);
    } catch (const std::exception& e) {
        std::cerr << "Could not load the BoC model: " << e.what() << "\n";
        return 1;
    }

    // The floor model is optional here: replaying BoC crops needs only the BoC
    // model, and the floor model may not be trained yet.
    std::unique_ptr<FloorDetector> floor;
    try {
        floor = std::make_unique<FloorDetector>(constants::floor_model_path,
                                                constants::img_width, constants::img_height);
    } catch (const std::exception& e) {
        std::cerr << "[main] floor model unavailable (" << constants::floor_model_path
                  << "); replaying BoC-only.\n";
    }

    RecipeSuggestor suggestor(seed);
    if (!start_seeds.empty()) suggestor.set_start_seeds(start_seeds);
    try {
        suggestor.bind_models(boc->class_names(),
                              floor ? floor->class_names() : std::vector<std::string>{});
    } catch (const std::exception& e) {
        std::cerr << "[main] class map mismatch: " << e.what() << "\n";
        return 1;
    }

    Router router;

    std::vector<fs::path> images;
    for (const auto& entry : fs::directory_iterator(dir)) {
        if (!entry.is_regular_file()) continue;
        std::string ext = entry.path().extension().string();
        if (ext == ".jpg" || ext == ".jpeg" || ext == ".png") images.push_back(entry.path());
    }
    std::sort(images.begin(), images.end());

    if (images.empty()) {
        std::cerr << "No images found in " << dir << "\n";
        return 1;
    }

    for (const auto& path : images) {
        cv::Mat frame = cv::imread(path.string());
        if (frame.empty()) {
            std::cerr << "Could not read " << path << "\n";
            continue;
        }

        std::cout << "\n=== " << path.filename().string() << " (" << frame.cols << "x" << frame.rows << ") ===\n";

        // A full-resolution frame is a whole screenshot: split it into the bag
        // region and the floor region exactly as the live pipeline does. A
        // 640x640 image is already a bag crop (that is what the dataset test
        // images are), so it goes straight to the BoC detector.
        const bool full_screen = frame.cols > constants::img_width && floor;

        cv::Mat boc_frame = frame;
        std::map<types::ConsumableID, types::Quantity> floor_supply;

        if (full_screen) {
            ScreenCapture source{nullptr, static_cast<u_int16_t>(frame.cols),
                                 static_cast<u_int16_t>(frame.rows)};
            const size_t bytes = static_cast<size_t>(frame.cols) * frame.rows * 3;
            source.data = new unsigned char[bytes];
            if (frame.isContinuous()) {
                std::memcpy(source.data, frame.data, bytes);
            } else {
                for (int y = 0; y < frame.rows; ++y) {
                    std::memcpy(source.data + static_cast<size_t>(y) * frame.cols * 3,
                                frame.ptr(y), static_cast<size_t>(frame.cols) * 3);
                }
            }

            ScreenCapture floor_img{nullptr, 0, 0};
            ScreenCapture boc_img{nullptr, 0, 0};
            router.route(&source, &floor_img, &boc_img);

            // route() can leave a region unfilled; screen_capture_to_mat would
            // memcpy from a null pointer, so check before converting -- the live
            // workers guard the same way.
            cv::Mat floor_mat;
            if (floor_img.data) floor_mat = screen_capture_to_mat(floor_img);
            if (!floor_mat.empty()) {
                std::vector<Prediction> floor_preds;
                floor_supply = floor->detect_floor(floor_mat, &floor_preds);

                (void)floor_preds;  // the report below names what was found
            }

            // screen_capture_to_mat copies, so the Mat outlives boc_img.
            if (boc_img.data) {
                cv::Mat routed_boc = screen_capture_to_mat(boc_img);
                if (!routed_boc.empty()) boc_frame = routed_boc;
            }
        }

        std::vector<Prediction> preds;
        auto detected = boc->detect(boc_frame, &preds);

        // Same reporting as the live pipeline: plans when the bag can be
        // filled, otherwise the state and the reason it cannot.
        auto suggestions = suggestor.suggest(detected.snapshot(), floor_supply);
        if (suggestions.empty()) {
            std::cout << suggestor.format_state(detected.snapshot(), floor_supply);
        } else {
            std::cout << suggestor.format(suggestions);
        }
    }
    return 0;
}

} // namespace

int main(int argc, char** argv) {
    std::string seed;
    std::string replay_dir;
    std::vector<uint32_t> start_seeds;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seed" && i + 1 < argc) {
            seed = argv[++i];
        } else if (arg == "--replay" && i + 1 < argc) {
            replay_dir = argv[++i];
        } else if (arg == "--start-seed" && i + 1 < argc) {
            start_seeds = parse_start_seeds(argv[++i]);
            if (start_seeds.empty()) {
                std::cerr << "No usable start seed in " << argv[i] << "\n";
                return 2;
            }
        } else if (arg == "--help" || arg == "-h") {
            usage(argv[0]);
            return 0;
        } else {
            std::cerr << "Unknown or incomplete argument: " << arg << "\n";
            usage(argv[0]);
            return 1;
        }
    }

    if (seed.empty()) {
        std::cerr << "[main] No --seed given; crafting results depend on the run seed.\n";
    }

    if (!replay_dir.empty()) {
        return run_replay(replay_dir, seed, start_seeds);
    }

    std::signal(SIGINT, handle_sigint);
    std::signal(SIGTERM, handle_sigint);

    std::pair<int, int> boc_shape = {constants::img_width, constants::img_height};
    std::pair<int, int> floor_shape = {constants::img_width, constants::img_height};

    auto pipeline = std::make_unique<Pipeline>(boc_shape, floor_shape);
    pipeline->initialize();
    pipeline->set_seed(seed);
    if (!start_seeds.empty()) pipeline->set_start_seeds(start_seeds);

    pipeline->run();
    std::cout << "[main] Running. Press Ctrl-C to stop.\n";

    while (pipeline->is_running() && !g_interrupted) {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    pipeline->stop();
    pipeline->join();
    std::cout << "[main] Stopped cleanly.\n";
    return 0;
}
