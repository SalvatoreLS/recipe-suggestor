#include <iostream>
#include <thread>
#include <chrono>
#include "pipeline/pipeline.hpp"
#include "constants.hpp"

int main() {
    std::cout << "Testing Pipeline..." << std::endl;

    std::pair<int, int> boc_shape = {constants::img_width, constants::img_height};
    std::pair<int, int> floor_shape = {constants::img_width, constants::img_height};

    Pipeline pipeline(boc_shape, floor_shape);
    pipeline.initialize();

    if (!pipeline.is_initialized()) {
        std::cerr << "Pipeline failed to initialize." << std::endl;
        return 1;
    }
    std::cout << "Pipeline initialized successfully." << std::endl;

    // run() spawns and returns, so the test thread is free to time the run.
    pipeline.run();
    if (!pipeline.is_running()) {
        std::cerr << "Pipeline did not start." << std::endl;
        return 1;
    }

    std::this_thread::sleep_for(std::chrono::seconds(1));

    std::cout << "Stopping Pipeline..." << std::endl;
    auto t0 = std::chrono::steady_clock::now();
    pipeline.stop();
    pipeline.join();
    auto elapsed = std::chrono::steady_clock::now() - t0;

    if (pipeline.is_running()) {
        std::cerr << "Pipeline still reports running after stop()." << std::endl;
        return 1;
    }

    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    std::cout << "Shutdown took " << ms << " ms." << std::endl;
    if (ms > 2000) {
        std::cerr << "Shutdown took too long (expected < 2000 ms)." << std::endl;
        return 1;
    }

    std::cout << "Pipeline test complete." << std::endl;
    return 0;
}
