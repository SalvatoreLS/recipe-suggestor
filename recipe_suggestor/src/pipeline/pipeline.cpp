#include "pipeline/pipeline.hpp"
#include <iostream>
#include <chrono>

Pipeline::Pipeline(std::pair<int, int> boc_shape, std::pair<int, int> floor_shape)
    : boc_shape_(boc_shape), floor_shape_(floor_shape) {
    this->initialize();
}

void Pipeline::initialize() {
    if (this->is_running() || this->is_initialized()) {
        return;
    }

    this->capturer = new FrameCapturer();
    this->router = new Router();
    this->suggestor = new RecipeSuggestor();

    // A detector whose model file is missing or unreadable must not take the
    // whole application down -- the ONNX Runtime exception used to propagate
    // straight out of the constructor and abort main().
    try {
        this->floor_detector = new FloorDetector(constants::floor_model_path,
                                                 floor_shape_.first, floor_shape_.second);
    } catch (const std::exception& e) {
        std::cerr << "[Pipeline] floor model unavailable (" << constants::floor_model_path
                  << "): " << e.what() << "\n";
        this->floor_detector = nullptr;
    }

    try {
        this->b_detector = new BoCDetector(constants::boc_model_path,
                                           boc_shape_.first, boc_shape_.second);
    } catch (const std::exception& e) {
        std::cerr << "[Pipeline] BoC model unavailable (" << constants::boc_model_path
                  << "): " << e.what() << "\n";
        this->b_detector = nullptr;
    }

    if (!this->floor_detector && !this->b_detector) {
        std::cerr << "[Pipeline] no detector could be loaded; nothing to do.\n";
    } else if (!this->floor_detector) {
        std::cout << "[Pipeline] running BoC-only (no floor detection).\n";
    } else if (!this->b_detector) {
        std::cout << "[Pipeline] running floor-only (no bag detection).\n";
    }

    // Cross-check the class maps against what the models actually report. Both
    // models are nc:21, so a swapped file is otherwise undetectable.
    try {
        this->suggestor->bind_models(
            this->b_detector ? this->b_detector->class_names() : std::vector<std::string>{},
            this->floor_detector ? this->floor_detector->class_names() : std::vector<std::string>{});
    } catch (const std::exception& e) {
        std::cerr << "[Pipeline] class map mismatch: " << e.what() << "\n";
    }

    std::cout << "[Pipeline] Initialized.\n";
    this->initialized = true;
}

// Workers (Nodes) ==================================================

void Pipeline::capture_worker() {
    while (this->is_running()) {
        ScreenCapture* frame = this->capturer->capture_screen();
        if (frame) {
            frame_queue.push(frame);
        } else {
            frame_queue.push(std::nullopt); // It tells the consumers to stop (end of queue)
            break;
        }
    }
}

void Pipeline::router_worker() {
    while (true) {
        auto msg = frame_queue.pop();
        if (!msg) {
            floor_image_queue.push(std::nullopt);
            boc_image_queue.push(std::nullopt);
            break;
        }

        ScreenCapture* source = *msg;
        ScreenCapture* floor_img = new ScreenCapture{nullptr, 0, 0};
        ScreenCapture* boc_img = new ScreenCapture{nullptr, 0, 0};

        this->router->route(source, floor_img, boc_img);

        floor_image_queue.push(floor_img);
        boc_image_queue.push(boc_img);

        delete source;
    }
}

void Pipeline::floor_worker() {
    while (true) {
        auto msg = floor_image_queue.pop();
        if (!msg) break;

        ScreenCapture* img = *msg;
        if (img && img->data && this->floor_detector) {
            cv::Mat mat = screen_capture_to_mat(*img);
            if (!mat.empty()) {
                auto results = this->floor_detector->detect_floor(mat);
                std::lock_guard<std::mutex> lock(results_mtx);
                this->shared_floor_obj = std::move(results);
            }
        }
        delete img;
    }
}

void Pipeline::boc_worker() {
    while (true) {
        auto msg = boc_image_queue.pop();
        if (!msg) break;

        ScreenCapture* img = *msg;
        if (img && img->data && this->b_detector) {
            cv::Mat mat = screen_capture_to_mat(*img);
            if (!mat.empty()) {
                auto results = this->b_detector->detect(mat);
                std::lock_guard<std::mutex> lock(results_mtx);
                this->shared_boc = std::move(results);
            }
        }
        delete img;
    }
}

void Pipeline::suggestor_worker() {
    while (this->is_running()) {
        // Snapshot under the lock, then do the work outside it: holding
        // results_mtx for the whole computation would stall both detectors.
        std::vector<types::ConsumableID> bag;
        std::map<types::ConsumableID, types::Quantity> floor;
        {
            std::lock_guard<std::mutex> lock(results_mtx);
            bag = this->shared_boc.snapshot();
            floor = this->shared_floor_obj;
        }

        auto suggestions = this->suggestor->suggest(bag, floor);
        if (!suggestions.empty()) {
            std::cout << this->suggestor->format(suggestions) << std::flush;
            last_state_line_.clear();
        } else {
            // No recipe is the normal case, not a failure: print what the
            // detectors see so the console shows the pipeline is alive, but
            // only when it changes, or it scrolls several times a second.
            std::string state = this->suggestor->format_state(bag, floor);
            if (state != last_state_line_) {
                std::cout << state << std::flush;
                last_state_line_ = std::move(state);
            }
        }

        // 30 suggestion blocks a second is unreadable, and the memo cache makes
        // the extra passes pointless anyway.
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }
}

// Execution Control ===============================================

void Pipeline::run() {
    if (this->is_running() || !this->is_initialized()) return;

    this->running = true;
    std::cout << "[Pipeline] Running Parallelized...\n";

    workers_.emplace_back(&Pipeline::capture_worker, this);
    workers_.emplace_back(&Pipeline::router_worker, this);
    workers_.emplace_back(&Pipeline::floor_worker, this);
    workers_.emplace_back(&Pipeline::boc_worker, this);
    workers_.emplace_back(&Pipeline::suggestor_worker, this);
}

void Pipeline::join() {
    for (auto& t : workers_) {
        if (t.joinable()) t.join();
    }
    workers_.clear();
}

void Pipeline::stop() {
    if (!this->running.exchange(false)) return;
    std::cout << "[Pipeline] Stopping...\n";

    // Closing the queues unblocks every worker parked in push() or pop().
    // Clearing `running` alone is not enough: capture_worker can be blocked in
    // push() on a full queue, and the three consumers block in pop() forever.
    frame_queue.close();
    floor_image_queue.close();
    boc_image_queue.close();

    // Closing is not enough on its own: pop() keeps handing out whatever is
    // still buffered, so the detectors would run inference over the entire
    // backlog (up to queue_max_size frames each) before reaching the sentinel --
    // several seconds of pointless work on the way out. Throw the backlog away
    // instead; the queues hold raw pointers, so free them here.
    for (ScreenCapture* leftover : frame_queue.drain())       delete leftover;
    for (ScreenCapture* leftover : floor_image_queue.drain()) delete leftover;
    for (ScreenCapture* leftover : boc_image_queue.drain())   delete leftover;
}

void Pipeline::set_seed(const std::string& seed) {
    if (this->suggestor) this->suggestor->set_seed(seed);
}

void Pipeline::set_start_seed(uint32_t seed) {
    if (this->suggestor) this->suggestor->set_start_seed(seed);
}

void Pipeline::set_start_seeds(std::vector<uint32_t> seeds) {
    if (this->suggestor) this->suggestor->set_start_seeds(std::move(seeds));
}

bool Pipeline::is_initialized() { return this->initialized; }
bool Pipeline::is_running() { return this->running; }

void Pipeline::drain_queues() {
    for (auto* p : frame_queue.drain()) delete p;
    for (auto* p : floor_image_queue.drain()) delete p;
    for (auto* p : boc_image_queue.drain()) delete p;
}

Pipeline::~Pipeline() {
    // Order matters: the workers must be gone before the nodes they use are
    // destroyed, otherwise a thread sitting inside session.Run() is left with
    // a dangling detector.
    this->stop();
    this->join();
    this->drain_queues();

    this->initialized = false;
    delete this->capturer;
    delete this->router;
    delete this->floor_detector;
    delete this->b_detector;
    delete this->suggestor;
}