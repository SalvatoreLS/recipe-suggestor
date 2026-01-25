#include "pipeline/nodes/floor_detector.hpp"

#include <map>

std::map<types::ConsumableID, types::Quantity> FloorDetector::detect_floor(const cv::Mat& frame, std::vector<Prediction>* out_predictions) {
    try {
        // 1. Preprocess
        cv::Mat blob = _preprocess_image(frame);
        
        // 2. Prepare Input Tensor
        size_t inputTensorSize = 1 * 3 * img_width * img_height;
        std::vector<float> inputTensorValues(inputTensorSize);
        
        if (blob.isContinuous()) {
            memcpy(inputTensorValues.data(), blob.ptr<float>(), inputTensorSize * sizeof(float));
        } else {
             // Fallback
             std::cerr << "[WARN] Blob not continuous, copy might fail or be slow.\n";
             // clone to make continuous
             cv::Mat cont = blob.clone();
             memcpy(inputTensorValues.data(), cont.ptr<float>(), inputTensorSize * sizeof(float));
        }
        
        // Create Tensor
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto inputTensor = Ort::Value::CreateTensor<float>(memoryInfo, 
                                                           inputTensorValues.data(), 
                                                           inputTensorSize, 
                                                           inputNodeDims.data(), 
                                                           inputNodeDims.size());

        // 3. Inference
        const char* inputNames[] = { inputName.c_str() };
        const char* outputNames[] = { outputName.c_str() };
        
        auto outputTensors = session.Run(Ort::RunOptions{nullptr}, inputNames, &inputTensor, 1, outputNames, 1);
        
        // 4. Process Output
        float* floatArr = outputTensors[0].GetTensorMutableData<float>();
        auto outputInfo = outputTensors[0].GetTensorTypeAndShapeInfo();
        std::vector<int64_t> outputShape = outputInfo.GetShape();
        size_t outputCount = outputInfo.GetElementCount();

        #if defined(DEBUG)
        std::cout << "\n=== DEBUG INFO ===" << std::endl;
        std::cout << "Output shape: [";
        for (size_t i = 0; i < outputShape.size(); i++) {
            std::cout << outputShape[i];
            if (i < outputShape.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Total elements: " << outputCount << std::endl;
        std::cout << "First 20 output values: ";
        for (int i = 0; i < std::min(20, (int)outputCount); i++) {
            std::cout << floatArr[i] << " ";
        }
        std::cout << std::endl;

        // Check max confidence across all predictions
        if (outputShape.size() == 3) {
            int64_t dimensions = outputShape[1];
            int64_t rows = outputShape[2];
            float max_conf_found = 0.0f;
            
            if (dimensions < rows) { // YOLOv8 format
                int num_classes = dimensions - 4;
                for (int i = 0; i < rows; i++) {
                    for (int c = 0; c < num_classes; c++) {
                        float score = floatArr[(4 + c) * rows + i];
                        max_conf_found = std::max(max_conf_found, score);
                    }
                }
            }
            std::cout << "Max confidence found in output: " << max_conf_found << std::endl;
        }
        std::cout << "==================\n" << std::endl;
        #endif
        
        std::vector<float> outputData(floatArr, floatArr + outputCount);

        #ifdef DEBUG
        std::cout << "\n=== DETAILED OUTPUT ANALYSIS ===" << std::endl;
        int64_t dim1 = outputShape[1];  // 25
        int64_t dim2 = outputShape[2];  // 8400

        // Check first detection's all 25 values
        std::cout << "First detection (all 25 channels):" << std::endl;
        for (int c = 0; c < dim1; c++) {
            std::cout << "  Channel " << c << ": " << floatArr[c * dim2 + 0] << std::endl;
        }

        // Check if confidences are in different positions
        std::cout << "\nChecking detection at index 100:" << std::endl;
        for (int c = 0; c < dim1; c++) {
            std::cout << "  Channel " << c << ": " << floatArr[c * dim2 + 100] << std::endl;
        }
        std::cout << "==================\n" << std::endl;
        #endif

        std::vector<Prediction> predictions = _filter_predictions(outputData, outputShape, frame.size());
        
        if (out_predictions) *out_predictions = predictions;
        
        std::map<types::ConsumableID, types::Quantity> detected_items;
        for (const auto& pred : predictions) detected_items[pred.classId]++;
        
        return detected_items;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] ONNX Runtime exception in detect: " << e.what() << std::endl;
        return std::map<types::ConsumableID, types::Quantity>();
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] OpenCV exception in detect: " << e.what() << std::endl;
        return std::map<types::ConsumableID, types::Quantity>();
    }
}