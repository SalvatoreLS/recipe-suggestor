#include "pipeline/pipeline.hpp"
#include <iostream>

Pipeline::Pipeline(std::pair<int, int> boc_shape, std::pair<int, int> floor_shape)
    : boc_shape_(boc_shape), floor_shape_(floor_shape) {
}

void Pipeline::initialize() {
    std::cout << "[Pipeline] Initialized." << std::endl;
}

void Pipeline::run() {
    std::cout << "[Pipeline] Running..." << std::endl;
}