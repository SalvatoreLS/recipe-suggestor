#include <onnxruntime_cxx_api.h>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <algorithm>

using namespace cv;
using namespace std;

// Helper to print first few values of output
void printOutputSummary(const std::vector<float>& output, const std::vector<int64_t>& shape) {
    long total_elements = 1;
    cout << "Output shape: [";
    for (size_t i = 0; i < shape.size(); ++i) {
        cout << shape[i] << (i < shape.size() - 1 ? ", " : "");
        total_elements *= shape[i];
    }
    cout << "]" << endl;

    cout << "First 5 values: ";
    for (int i = 0; i < std::min((long)5, total_elements); ++i) {
        cout << output[i] << " ";
    }
    cout << endl;
}

int main(int argc, char** argv) {
    string modelPath = (argc > 1) ? argv[1] : "best.onnx";
    cout << "Attempting to load model from: " << modelPath << " using ONNX Runtime" << endl;

    try {
        // 1. Initialize Environment
        Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "YOLOv8_Inference");

        // 2. Session Options
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);

        // 3. Create Session
        Ort::Session session(env, modelPath.c_str(), sessionOptions);
        cout << "Model loaded successfully!" << endl;

        // 4. Prepare Input Data (using OpenCV for image loading/preprocessing)
        // Dummy image 640x640
        Mat frame = Mat::zeros(640, 640, CV_8UC3);
        
        // Preprocessing: YOLOv8 expects RGB, 0-1 normalized, (1, 3, 640, 640)
        Mat blob;
        dnn::blobFromImage(frame, blob, 1.0/255.0, Size(640, 640), Scalar(), true, false);

        // Convert cv::Mat (NCHW float32) to vector<float>
        size_t inputTensorSize = 1 * 3 * 640 * 640;
        vector<float> inputTensorValues(inputTensorSize);
        // blobFromImage output is continuous, so we can copy directly
        if (blob.isContinuous()) {
            memcpy(inputTensorValues.data(), blob.ptr<float>(), inputTensorSize * sizeof(float));
        } else {
             // Fallback if not continuous (unlikely for blobFromImage)
             cout << "Warning: Blob is not continuous, manual copy." << endl;
             // implementation left out for brevity as blobFromImage is usually continuous
        }

        // 5. Define Input/Output Names and Shapes
        Ort::AllocatorWithDefaultOptions allocator;
        
        // Input info
        auto inputNamePtr = session.GetInputNameAllocated(0, allocator);
        string inputName = inputNamePtr.get();
        vector<int64_t> inputNodeDims = {1, 3, 640, 640};

        // Output info
        auto outputNamePtr = session.GetOutputNameAllocated(0, allocator);
        string outputName = outputNamePtr.get();

        // 6. Create Input Tensor
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, inputTensorValues.data(), inputTensorSize, inputNodeDims.data(), inputNodeDims.size());

        // 7. Run Inference
        cout << "Running inference..." << endl;
        const char* inputNames[] = { inputName.c_str() };
        const char* outputNames[] = { outputName.c_str() };
        
        auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
        cout << "Inference successful." << endl;

        // 8. Process Output
        float* floatarr = outputTensors[0].GetTensorMutableData<float>();
        auto outputInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
        vector<int64_t> outputShape = outputInfo.GetShape();
        
        // Copy to vector for safer handling/printing
        size_t outputSize = outputInfo.GetElementCount();
        vector<float> outputData(floatarr, floatarr + outputSize);

        printOutputSummary(outputData, outputShape);

    } catch (const Ort::Exception& e) {
        cerr << "ONNX Runtime Exception: " << e.what() << endl;
        return -1;
    } catch (const std::exception& e) {
        cerr << "Exception: " << e.what() << endl;
        return -1;
    }

    return 0;
}
