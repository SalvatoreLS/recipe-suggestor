# RecipeSuggestorCPP

C++ implementation of the Recipe Suggestor pipeline using **ONNX Runtime** for efficient object detection.

## Prerequisites

- **CMake** 3.10+
- **OpenCV** 4.x
- **ONNX Runtime** (libonnxruntime)

## Compilation Instructions

Standard CMake build process:

```bash
mkdir build
cd build
cmake ..
make
```

> The build system attempts to locate `onnxruntime` automatically. It may use a local copy in `models_training/cpp_compatibility_test/lib` if configured, or a system installation. The shared library `libonnxruntime.so` is copied to the build directory for runtime linkage.

## Running the Application

After compilation, run the main executable:

```bash
./RecipeSuggestorCPP
```

## Running Tests

To verify the detection pipeline:

```bash
./detector_test
```

### Test Resources
The test expects the following file structure:
- Model: `resources/models/best.onnx`
- Test Image: `resources/test_images/test1.jpg`
- Output: `outputs/detector_test_output.jpg`

## Project Structure
- `src/`: Source files.
- `include/`: Header files.
- `tests/`: Unit and integration tests.
- `resources/`: Models and test data.