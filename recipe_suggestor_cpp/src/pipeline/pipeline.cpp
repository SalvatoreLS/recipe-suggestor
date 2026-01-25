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
    
    // Default model path
    std::string model_path = "resources/models/best.onnx"; 

    this->capturer = new FrameCapturer();
    this->router = new Router();
    this->floor_detector = new FloorDetector(model_path, floor_shape_.first, floor_shape_.second);
    this->b_detector = new BoCDetector(model_path, boc_shape_.first, boc_shape_.second);
    this->suggestor = new RecipeSuggestor();

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
            frame_queue.push(std::nullopt); // Poison pill
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
        if (img && img->data) {
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
        if (img && img->data) {
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
        {
            std::lock_guard<std::mutex> lock(results_mtx);
            this->suggestor->suggest(this->shared_boc, this->shared_floor_obj);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(33));
    }
}

// Execution Control ===============================================

void Pipeline::run() {
    if (this->is_running() || !this->is_initialized()) return;

    this->running = true;
    std::cout << "[Pipeline] Running Parallelized...\n";

    std::vector<std::thread> workers;
    workers.emplace_back(&Pipeline::capture_worker, this);
    workers.emplace_back(&Pipeline::router_worker, this);
    workers.emplace_back(&Pipeline::floor_worker, this);
    workers.emplace_back(&Pipeline::boc_worker, this);
    workers.emplace_back(&Pipeline::suggestor_worker, this);

    for (auto& t : workers) {
        if (t.joinable()) t.join();
    }
}

bool Pipeline::is_initialized() { return this->initialized; }
bool Pipeline::is_running() { return this->running; }

Pipeline::~Pipeline() {
    this->running = false;
    delete this->capturer;
    delete this->router;
    delete this->floor_detector;
    delete this->b_detector;
    delete this->suggestor;
}