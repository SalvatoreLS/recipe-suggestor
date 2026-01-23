#include <sys/types.h>
#include "types.hpp"
#include <utility>
#include <atomic>
#include "nodes/boc_detector.hpp"
#include "nodes/floor_detector.hpp"
#include "nodes/frame_capturer.hpp"
#include "nodes/recipe_suggestor.hpp"
#include "nodes/router.hpp"
#include "utils.hpp"

class Pipeline {
public:
    Pipeline(std::pair<int, int> boc_shape, std::pair<int, int> floor_shape);
    void initialize();
    bool is_initialized();
    void run();
    bool is_running();
    ~Pipeline();

private:
    std::pair<int, int> boc_shape_;
    std::pair<int, int> floor_shape_;
    std::atomic<bool> running = false;
    std::atomic<bool> initialized = false;
    std::atomic<types::FrameCount> frame_count = 0; 
    
    // Threads-related objects
    FrameCapturer* capturer = nullptr;
    Router* router = nullptr;
    FloorDetector* floor_detector = nullptr;
    BoCDetector* boc_detector = nullptr;
    RecipeSuggestor* suggestor = nullptr;
};