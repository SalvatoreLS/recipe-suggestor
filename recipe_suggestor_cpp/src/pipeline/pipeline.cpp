#include "pipeline/pipeline.hpp"
#include <iostream>
#include <sys/types.h>
#include <thread>
#include <future>
#include <map>
#include "data_structures/circular_list.hpp"
#include "utils.hpp"
#include <opencv2/opencv.hpp>

Pipeline::Pipeline(std::pair<int, int> boc_shape, std::pair<int, int> floor_shape)
    : boc_shape_(boc_shape), floor_shape_(floor_shape) {
        // TODO: check if additional variables are needed
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

        // Capture frame
        ScreenCapture* curr_frame = this->capturer->capture_screen();
        if (!curr_frame) continue;

        ScreenCapture floor_img = {nullptr, 0, 0};
        ScreenCapture boc_img = {nullptr, 0, 0};
        
        // Process frame
        this->router->route(curr_frame, &floor_img, &boc_img);

        // Detect Floor
        if (floor_img.data) {
            cv::Mat floor_mat = screen_capture_to_mat(floor_img);
            if (!floor_mat.empty()) {
                this->floor_detector->detect(floor_mat);
            }
        }

        // Detect BoC
        if (boc_img.data) {
            cv::Mat boc_mat = screen_capture_to_mat(boc_img);
            if (!boc_mat.empty()) {
               this->boc_detector->detect(boc_mat);
            }
        }
        
        // TODO: define a data structure for the suggestion

        // Retrive suggestions
        this->suggestor->suggest(boc, floor_obj);

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