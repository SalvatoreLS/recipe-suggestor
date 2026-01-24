#include "pipeline/pipeline.hpp"
#include <iostream>
#include <sys/types.h>
#include <thread>
#include <future>
#include <map>
#include <optional>
#include "data_structures/circular_list.hpp"
#include "types.hpp"
#include "utils.hpp"
#include <opencv2/opencv.hpp>


/*
PIPELINE MESSAGES:
// Frame capturing : None           ==>   ScreenCapture
// Router          : ScreenCapture  ==>   [ ScreenCapture, ScreenCapture ]
// [ PROCESS FRAMES ScreenCapture => cv::Mat ]
// BOCDetector     : ScreenCapture  ==>   cust::CircularList<types::ConsumableID>
// FloorDetector   : ScreenCapture  ==>   map<types::ConsumableID>
// RecipeSuggestor : [ cust::CircularList<types::ConsumableID>, unordered_map<types::ConsumableID> ] => TBD
*/

// PIPELINE NODES ==================================================

std::optional<ScreenCapture*> Pipeline::frame_node() {
    // TODO: ADD PARALLELISM
    ScreenCapture* frame = this->capturer->capture_screen();
    if (!frame) return std::nullopt;
    return frame;
}

void router_node(ScreenCapture* source_img, ScreenCapture* floor_img, ScreenCapture* boc_img) {
    // TODO: ADD PARALLELISM
    this->router->route(source_img, floor_img, boc_img);
}

std::map<types::ItemID, types::Quantity> Pipeline::floor_node(cv::Mat& floor_mat) {
    // TODO: ADD PARALLELISM
    return this->floor_detector->detect_floor(floor_mat);
}

cust::CircularList<types::ItemID> boc_node(cv::Mat& boc_mat) {
    // TODO: ADD PARALLELISM
    return this->boc_detector->detect(boc_mat);
}

// =================================================================

Pipeline::Pipeline(std::pair<int, int> boc_shape, std::pair<int, int> floor_shape)
    : boc_shape_(boc_shape), floor_shape_(floor_shape) {
    this->initialize();
}

void Pipeline::initialize() {
    if (this->is_running()) {
        std::cout << "[Pipeline] Trying to initialize when pipeline is running\n";
        return;
    }

    if (this->is_initialized()) {
        std::cout << "[Pipeline] Already initialized.\n";
        return;
    }
    
    // Default model path
    std::string model_path = "resources/models/best.onnx"; 

    this->capturer = new FrameCapturer();
    this->router = new Router();
    this->floor_detector = new FloorDetector(model_path, floor_shape_.first, floor_shape_.second);
    this->boc_detector = new BoCDetector(model_path, boc_shape_.first, boc_shape_.second);
    this->suggestor = new RecipeSuggestor();

    std::cout << "[Pipeline] Initialized.\n";
    this->initialized = true;
}

bool Pipeline::is_initialized() { return this->initialized; }

void Pipeline::run() {
    if (this->is_running()) {
        std::cout << "[Pipeline] Already running.\n";
        return;
    }

    if (!this->is_initialized()) {
        std::cout << "[Pipeline] Trying to run pipeline before initialization.\n";
        return;
    }

    this->running = true;
    std::cout << "[Pipeline] Running...\n";

    // Structures for detections
    cust::CircularList<types::ItemID> boc;
    std::map<types::ItemID, types::Quantity> floor_obj; // (id, quantity)
    
    while (this->is_running()) {
        ++this->frame_count;
        std::cout << "[Pipeline] Processing frame " << this->frame_count << "\n";

        // Capture frame ==========================================

        std::optional<ScreenCapture*> curr_frame = this->frame_node();
        if (!curr_frame) continue;

        // ========================================================

        // Route frame ============================================
        
        ScreenCapture floor_img = {nullptr, 0, 0};
        ScreenCapture boc_img = {nullptr, 0, 0};

        this->router_node(curr_frame, &floor_img, &boc_img);

        // ========================================================

        // Detect Floor ===========================================

        cust::CircularList<types::ConsumableID> floor_obj; // TODO: redefine detect method in floor detector to make it return data as a map
                                                     // (call Detector::detect() and convert)

                                                     // TODO: define an order for CircularList and always return items using the same criteria (rotate till correct before return)
    
        if (floor_img.data) {
            cv::Mat floor_mat = screen_capture_to_mat(floor_img);
            if (!floor_mat.empty()) floor_obj = this->floor_node(floor_mat);
        }

        // ========================================================

        // Detect BoC =============================================

        cust::CircularList<types::ConsumableID> boc_obj;
        if (boc_img.data) {
            cv::Mat boc_mat = screen_capture_to_mat(boc_img);
            if (!boc_mat.empty()) boc_obj = this->boc_node(boc_mat);
        }

        // ========================================================
        
        // Retrive suggestions ====================================
        
        // TODO: define a data structure for the suggestions (Trie + ranking (??))

        this->suggestor->suggest(boc, floor_obj);

        // ========================================================

        delete curr_frame;
    }
}

bool Pipeline::is_running() { return this->running; }

Pipeline::~Pipeline() {
    this->running = false;
    delete this->capturer;
    delete this->router;
    delete this->floor_detector;
    delete this->boc_detector;
    delete this->suggestor;
}