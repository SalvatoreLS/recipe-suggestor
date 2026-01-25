#include "data_structures/circular_list.hpp"
#include "types.hpp"
#include "detector.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

Detector::Detector(const std::string& model_path, int wid, int hei) 
    : img_width(wid), img_height(hei), 
      env(ORT_LOGGING_LEVEL_WARNING, "RecipeSuggestorDetector"),
      session(nullptr) {
    
    try {
        Ort::SessionOptions sessionOptions;
        sessionOptions.SetIntraOpNumThreads(constants::intra_op_num_threads);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        
        session = Ort::Session(env, model_path.c_str(), sessionOptions);
        
        // Input
        size_t numInputNodes = session.GetInputCount();
        if (numInputNodes > 0) {
            auto inputNamePtr = session.GetInputNameAllocated(0, allocator);
            inputName = inputNamePtr.get();
            
            auto typeInfo = session.GetInputTypeInfo(0);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            inputNodeDims = tensorInfo.GetShape();
            
            // Handle negative/dynamic dims if necessary
            if (inputNodeDims.size() == 4) {
                 if (inputNodeDims[2] == -1 || inputNodeDims[3] == -1) {
                     inputNodeDims[2] = img_height;
                     inputNodeDims[3] = img_width;
                 }
            }
        }

        // Output
        size_t numOutputNodes = session.GetOutputCount();
        if (numOutputNodes > 0) {
            auto outputNamePtr = session.GetOutputNameAllocated(0, allocator);
            outputName = outputNamePtr.get();
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] ONNX Runtime exception: " << e.what() << "\n";
        throw;
    }
}

cv::Mat Detector::_preprocess_image(const cv::Mat& frame) {
    cv::Mat blob;
    cv::dnn::blobFromImage(
        frame,                          // input image (source)
        blob,                           // output blob (destination)
        constants::pixel_scale,         // scale factor (in constants.hpp)
        cv::Size(img_width, img_height),
        cv::Scalar(0, 0, 0),
        true,                           // swapRB (BGR->RGB)
        false                           // crop
    );

    #if defined(DEBUG)
    std::cout << "[DEBUG from _preprocess_image] Preprocessed image to blob." << std::endl;
    std::cout << "Blob shape: [" << blob.size[0] << ", " 
              << blob.size[1] << ", " 
              << blob.size[2] << ", " 
              << blob.size[3] << "]" << std::endl;
    std::cout << "Blob value range: [" 
              << *std::min_element(blob.begin<float>(), blob.end<float>()) << ", "
              << *std::max_element(blob.begin<float>(), blob.end<float>()) << "]" << std::endl;
    #endif

    return blob;
}

std::vector<Prediction> Detector::_filter_predictions(const std::vector<float>& output, const std::vector<int64_t>& shape, const cv::Size& img_size, float conf_threshold) {
    std::vector<Prediction> preds;
    if (shape.size() < 3) return preds;

    int64_t dimensions = shape[1];
    int64_t rows = shape[2];
        
    if (dimensions < rows && dimensions > 4) {
        int num_classes = dimensions - 4;
        const float* data = output.data();
        
        std::vector<cv::Rect> bboxes;
        std::vector<float> scores;
        std::vector<int> classIds;

        for (int i = 0; i < rows; ++i) {
             float max_score = 0.0f;
             int max_class_id = -1;
             
             for (int c = 0; c < num_classes; ++c) {
                 float score = data[(4 + c) * rows + i];
                 if (score > max_score) {
                     max_score = score;
                     max_class_id = c;
                 }
             }

             if (max_score >= conf_threshold) {
                 float cx = data[0 * rows + i];
                 float cy = data[1 * rows + i];
                 float w  = data[2 * rows + i];
                 float h  = data[3 * rows + i];

                 float scale_x = (float)img_size.width / img_width;
                 float scale_y = (float)img_size.height / img_height;
                 
                 int x = (int)((cx - 0.5 * w) * scale_x);
                 int y = (int)((cy - 0.5 * h) * scale_y);
                 int width = (int)(w * scale_x);
                 int height = (int)(h * scale_y);
                 
                 bboxes.push_back(cv::Rect(x, y, width, height));
                 scores.push_back(max_score);
                 classIds.push_back(max_class_id);
             }
        }
        
        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes, scores, conf_threshold, constants::nms_threshold, indices);
        
        for (int idx : indices) {
            preds.push_back({bboxes[idx], classIds[idx], scores[idx]});
        }
    } 
    
    return preds;
}

void Detector::_sort_bboxes(std::vector<Prediction>& predictions, int threshold) {
    std::sort(predictions.begin(), predictions.end(), [threshold](const Prediction& a, const Prediction& b) {
        if (abs(a.bbox.y - b.bbox.y) > threshold)
            return a.bbox.y < b.bbox.y;
        return a.bbox.x < b.bbox.x;
    });
}

cust::CircularList<types::ConsumableID> Detector::detect(const cv::Mat& frame, std::vector<Prediction>* out_predictions) {
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
        
        // Sort
        _sort_bboxes(predictions); // TO BE TESTED
        
        if (out_predictions) *out_predictions = predictions;
        
        cust::CircularList<types::ConsumableID> detected_items;
        for (const auto& pred : predictions) {
            detected_items.add(static_cast<types::ConsumableID>(pred.classId));
        }
        
        return detected_items;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] ONNX Runtime exception in detect: " << e.what() << std::endl;
        return cust::CircularList<types::ConsumableID>();
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] OpenCV exception in detect: " << e.what() << std::endl;
        return cust::CircularList<types::ConsumableID>();
    }
}



cv::Mat Detector::visualize_detections(const cv::Mat& frame, const std::vector<Prediction>& predictions, const std::vector<std::string>& class_names) {
    cv::Mat vis_image = frame.clone();
    for (const auto& pred : predictions) {
        cv::rectangle(vis_image, pred.bbox, cv::Scalar(0, 255, 0), 2);
        std::string class_name = (pred.classId < (int)class_names.size()) ? class_names[pred.classId] : std::to_string(pred.classId);
        std::string label = class_name + ": " + cv::format("%.2f", pred.confidence);
        
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(pred.bbox.y, labelSize.height);
        
        cv::rectangle(vis_image, cv::Point(pred.bbox.x, top - labelSize.height),
                      cv::Point(pred.bbox.x + labelSize.width, top + baseLine),
                      cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(vis_image, label, cv::Point(pred.bbox.x, top),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    return vis_image;
}