#pragma once
#include <utility>

class Pipeline {
public:
    Pipeline(std::pair<int, int> boc_shape, std::pair<int, int> floor_shape);
    void initialize();
    void run();

private:
    std::pair<int, int> boc_shape_;
    std::pair<int, int> floor_shape_;
};