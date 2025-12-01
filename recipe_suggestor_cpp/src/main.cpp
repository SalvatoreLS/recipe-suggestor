#include <iostream>
#include "pipeline/pipeline.hpp"

int main(void) {

  std::pair<int, int> boc_shape = {640, 640};
  std::pair<int, int> floor_shape = {640, 640};

  Pipeline* pipeline = new Pipeline(boc_shape, floor_shape);

  pipeline->initialize();

  pipeline->run();

  std::cout << "Started main" << std::endl;
  return 1;
}
