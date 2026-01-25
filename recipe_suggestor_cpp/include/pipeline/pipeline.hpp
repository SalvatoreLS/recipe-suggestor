#ifndef PIPELINE_HPP
#define PIPELINE_HPP


#include <sys/types.h>
#include <utility>
#include <atomic>
#include <thread>
#include <vector>
#include <mutex>
#include <map>


#include "types.hpp"
#include "nodes/boc_detector.hpp"
#include "nodes/floor_detector.hpp"
#include "nodes/frame_capturer.hpp"

// Undefine X11 macros to prevent conflicts with OpenCV/ONNX
#ifdef Status
#undef Status
#endif
#ifdef None
#undef None
#endif
#ifdef Success
#undef Success
#endif

#include "nodes/recipe_suggestor.hpp"
#include "nodes/router.hpp"
#include "utils.hpp"
#include "data_structures/bounded_queue.hpp"
#include "constants.hpp"

class Pipeline {
public:
    Pipeline(std::pair<int, int> boc_shape, std::pair<int, int> floor_shape);
    ~Pipeline();

    void initialize();
    bool is_initialized();
    void run();
    bool is_running();

private:
    // Config and state
    std::pair<int, int> boc_shape_;
    std::pair<int, int> floor_shape_;
    std::atomic<bool> running{false};
    std::atomic<bool> initialized{false};
    std::atomic<types::FrameCount> frame_count{0};

    // Processing queues 
    BoundedQueue<ScreenCapture*> frame_queue{constants::queue_max_size};
    BoundedQueue<ScreenCapture*> floor_image_queue{constants::queue_max_size};
    BoundedQueue<ScreenCapture*> boc_image_queue{constants::queue_max_size};

    // Thread-safe shared results
    std::mutex results_mtx;
    cust::CircularList<types::ItemID> shared_boc;
    std::map<types::ItemID, types::Quantity> shared_floor_obj;
    // TODO: Trie structure

    // Node objects
    FrameCapturer* capturer = nullptr;
    Router* router = nullptr;
    FloorDetector* floor_detector = nullptr;
    BoCDetector* b_detector = nullptr;
    RecipeSuggestor* suggestor = nullptr; // TODO: implement this class correctly

    // Worker Threads
    void capture_worker();
    void router_worker();
    void floor_worker();
    void boc_worker();
    void suggestor_worker();

    // Stop threads
    void stop();
};

#endif // PIPELINE_HPP