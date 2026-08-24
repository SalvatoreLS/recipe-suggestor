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

    // run() spawns the workers and returns immediately; join() waits for them.
    // Keeping them separate is what makes stop() callable from another thread
    // (or from a signal handler) instead of the process only ever being killed.
    void run();
    void join();
    void stop();
    bool is_running();

    void set_seed(const std::string& seed);
    void set_start_seed(uint32_t seed);
    void set_start_seeds(std::vector<uint32_t> seeds);

private:
    // Config and state
    std::pair<int, int> boc_shape_;
    std::pair<int, int> floor_shape_;
    std::atomic<bool> running{false};
    std::atomic<bool> initialized{false};

    // Processing queues 
    BoundedQueue<ScreenCapture*> frame_queue{constants::queue_max_size};
    BoundedQueue<ScreenCapture*> floor_image_queue{constants::queue_max_size};
    BoundedQueue<ScreenCapture*> boc_image_queue{constants::queue_max_size};

    // Thread-safe shared results
    std::mutex results_mtx;
    cust::CircularList<types::ConsumableID> shared_boc;
    std::map<types::ConsumableID, types::Quantity> shared_floor_obj;

    // Node objects
    FrameCapturer* capturer = nullptr;
    Router* router = nullptr;
    FloorDetector* floor_detector = nullptr;
    BoCDetector* b_detector = nullptr;
    RecipeSuggestor* suggestor = nullptr;

    // Last state line printed by suggestor_worker, so an unchanged game state
    // is not reprinted on every poll. Touched only by that one thread.
    std::string last_state_line_;

    // Worker threads, owned so that stop()/join() can reach them.
    std::vector<std::thread> workers_;

    // Worker Threads
    void capture_worker();
    void router_worker();
    void floor_worker();
    void boc_worker();
    void suggestor_worker();

    // Frees anything still sitting in the queues at shutdown.
    void drain_queues();
};

#endif // PIPELINE_HPP