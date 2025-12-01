## TODO list
- [x] Create class Detector
- [ ] Write unit tests for Detector class

## Compilation Instructions

This project supports two inference backends: OpenCV DNN (default) and LibTorch. The Detector class can use either based on compilation flags.

### Prerequisites
- CMake 3.10+
- OpenCV 4.x (with DNN module)
- For LibTorch: Download and extract LibTorch (CPU version) from https://pytorch.org/cppdist/. Place it in the `libtorch/` directory.

### Building the Main Program

#### Without LibTorch (OpenCV DNN)
```bash
mkdir build
cd build
cmake ..
make
```
This builds the main executable using OpenCV DNN for inference.

#### With LibTorch
```bash
mkdir build
cd build
cmake -DUSE_LIBTORCH=ON ..
make
```
This builds the main executable using LibTorch for inference. Ensure LibTorch is in `libtorch/`.

### Building Tests

#### Detector Test (OpenCV DNN)
Compile manually:
```bash
g++ -std=c++17 -Iinclude -I/usr/include/opencv4 \
    tests/detector_test.cpp \
    src/pipeline/detector.cpp \
    src/crafting_computation/crafting_computator.cpp \
    src/custom_onnx_import.cpp \
    -L/usr/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn \
    -o tests/detector_test
```

Run: `./tests/detector_test`

#### Detector Test (LibTorch)
When building with `-DUSE_LIBTORCH=ON`, the test is built automatically as `detector_test_torch`.

Compile manually if needed:
```bash
g++ -std=c++17 -DUSE_LIBTORCH -Iinclude -I/usr/include/opencv4 -Ilibtorch/include -Ilibtorch/include/torch/csrc/api/include \
    tests/detector_test_torch.cpp \
    src/pipeline/detector_torch.cpp \
    src/crafting_computation/crafting_computator.cpp \
    src/custom_onnx_import.cpp \
    -L/usr/lib -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_dnn \
    -Llibtorch/lib -ltorch -ltorch_cpu -lc10 \
    -o tests/detector_test_torch
```

To run the test, set the library path and execute:
```bash
export LD_LIBRARY_PATH=libtorch/lib:$LD_LIBRARY_PATH
./tests/detector_test_torch
```

### Running the Program
After building, run the main executable:
```bash
./RecipeSuggestorCPP
```

### Notes
- The OpenCV DNN version uses Darknet/YOLO models loaded via `cv::dnn::readNetFromDarknet`.
- The LibTorch version uses TorchScript models loaded via `torch::jit::load`.
- Ensure model files are in `resources/models/BoC_model/` and test images in `resources/test_images/`.
- Outputs are saved to `outputs/`.