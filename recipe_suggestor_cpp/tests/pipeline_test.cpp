#include <iostream>
#include <thread>
#include <chrono>
#include "pipeline/pipeline.hpp"
#include "constants.hpp"

int main() {
    std::cout << "Testing Pipeline..." << std::endl;
    
    std::pair<int, int> boc_shape = {constants::img_width, constants::img_height};
    std::pair<int, int> floor_shape = {constants::img_width, constants::img_height};
    
    // Create Pipeline
    Pipeline* pipeline = new Pipeline(boc_shape, floor_shape);
    
    // Initialize
    pipeline->initialize();
    
    if (pipeline->is_initialized()) {
        std::cout << "Pipeline initialized successfully." << std::endl;
    } else {
        std::cerr << "Pipeline failed to initialize." << std::endl;
        return 1;
    }
    
    // Run in a separate thread for a short duration
    std::thread pipeline_thread([&]() {
        pipeline->run();
    });
    
    // Let it run for 1 second
    std::this_thread::sleep_for(std::chrono::seconds(1));
    
    std::cout << "Stopping Pipeline..." << std::endl;
    delete pipeline;
    
    if (pipeline_thread.joinable())
        pipeline_thread.detach();

    std::cout << "Pipeline test complete." << std::endl;
    return 0;
}
