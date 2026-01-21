#include "data_structures/circular_list.hpp"
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
        sessionOptions.SetIntraOpNumThreads(1);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
        
        session = Ort::Session(env, model_path.c_str(), sessionOptions);
        
        // Resolve input/output names
        // Note: For complex models, we might need to iterate over inputs/outputs. 
        // Assuming single input/output for simplicity as per reference or YOLO style.
        
        // Input
        size_t numInputNodes = session.GetInputCount();
        if (numInputNodes > 0) {
            auto inputNamePtr = session.GetInputNameAllocated(0, allocator);
            inputName = inputNamePtr.get();
            
            auto typeInfo = session.GetInputTypeInfo(0);
            auto tensorInfo = typeInfo.GetTensorTypeAndShapeInfo();
            inputNodeDims = tensorInfo.GetShape();
            
            // Handle negative/dynamic dims if necessary, usually standard models have fixed input or -1 batch
            // Use constructor wid/hei if dims are dynamic
            if (inputNodeDims.size() == 4) {
                 if (inputNodeDims[2] == -1 || inputNodeDims[3] == -1) {
                     inputNodeDims[2] = img_height;
                     inputNodeDims[3] = img_width;
                 }
                 // Override with constructor args if different (resizing will happen in preprocess)
                 // But ideally model expects specific size.
                 // For now, trusting constructor args match model requirement or resize target.
            }
        }

        // Output
        size_t numOutputNodes = session.GetOutputCount();
        if (numOutputNodes > 0) {
            auto outputNamePtr = session.GetOutputNameAllocated(0, allocator);
            outputName = outputNamePtr.get();
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] ONNX Runtime exception: " << e.what() << std::endl;
        throw;
    }
}

cv::Mat Detector::_preprocess_image(const cv::Mat& frame) {
    cv::Mat blob;
    // YOLO typically expects RGB, normalized 0-1.
    // blobFromImage handles resizing, swapping BGR->RGB (if swapRB=true), and scaling.
    cv::dnn::blobFromImage(frame, blob,
        1.0 / 255.0,
        cv::Size(img_width, img_height),
        cv::Scalar(0, 0, 0),
        true, // swapRB
        false); // crop
    return blob;
}

std::vector<Prediction> Detector::_filter_predictions(const std::vector<float>& output, const std::vector<int64_t>& shape, const cv::Size& img_size, float conf_threshold) {
    std::vector<Prediction> preds;
    
    // YOLOv8/v5 ONNX output shape is typically [1, 84, 8400] (Batch, Class+Box, Anchors) or [1, 25200, 85] depending on export.
    // We need to handle the shape correctly.
    // Based on test_cpp, we are getting a flat vector.
    // Assuming shape is [1, dimensions, detections] or [1, detections, dimensions]
    
    if (shape.size() < 3) return preds;

    int64_t dimensions = shape[1];
    int64_t rows = shape[2];
    
    // Check if we need to transpose: YOLOv8 often outputs [1, 4+classes, N]
    bool is_yolov8_format = (dimensions < rows && dimensions > 4); 
    
    if (is_yolov8_format) {
        // dimensions = entries per anchor (cx, cy, w, h, class_scores...)
        // rows = number of anchors
        
        // Actually, YOLOv8 default export is [1, 84, 8400] where 84 is (cx, cy, w, h, 80 classes)
        // We need to iterate over the 8400 columns.
        
        int num_classes = dimensions - 4;
        const float* data = output.data();
        
        std::vector<cv::Rect> bboxes;
        std::vector<float> scores;
        std::vector<int> classIds;

        for (int i = 0; i < rows; ++i) {
             // Extract class scores
             float max_score = 0.0f;
             int max_class_id = -1;
             
             // Stride is rows (since it's column-major if we view it as [dim, rows] but usually data is linear C-order [batch, dim, rows])
             // wait, ONNX storage is row-major.
             // data[d * rows + i] ?? No, let's verify standard YOLOv8 output.
             // Usually it's [batch, channels, anchors].
             // So for a specific anchor 'i', attributes are at data[0*rows + i], data[1*rows + i]...
             
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

                 int x = static_cast<int>((cx - w / 2) * img_size.width / img_width); // scaling if input != original
                 int y = static_cast<int>((cy - h / 2) * img_size.height / img_height);
                 // Note: if img_width/height are the model input size, and we resize original image to it.
                 // We should scale bbox back to original image size.
                 // Here `img_size` is original frame size.
                 // cx, cy, w, h are usually in pixels of the defined input size (img_width, img_height).
                 // So we scale by (original / input_dim).
                 
                 float scale_x = (float)img_size.width / img_width;
                 float scale_y = (float)img_size.height / img_height;
                 
                 x = (int)((cx - 0.5 * w) * scale_x);
                 y = (int)((cy - 0.5 * h) * scale_y);
                 int width = (int)(w * scale_x);
                 int height = (int)(h * scale_y);
                 
                 bboxes.push_back(cv::Rect(x, y, width, height));
                 scores.push_back(max_score);
                 classIds.push_back(max_class_id);
             }
        }
        
        // NMS
        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes, scores, conf_threshold, 0.45f, indices);
        
        for (int idx : indices) {
            preds.push_back({bboxes[idx], classIds[idx], scores[idx]});
        }
        
    } else {
        // Fallback for [1, N, 85] (YOLOv5 style sometimes)
        // dimensions = N, rows = 85? No.
        // If shape is [1, 25200, 85] -> dimensions=25200, rows=85.
        // Handled differently.
        // Let's assume YOLOv8 format for now as that's the modern default and likely context.
        // If it's the other way around, we can add logic.
        // But the previous implementation logic handled "rows" as detections.
        
        // Re-using previous logic structure roughly?
        // Let's stick strictly to what ONNX usually delivers for YOLO.
        // If the user's model is YOLOv5, it might be [1, N, 85].
        
        // Let's look at `test_cpp.cpp`. It didn't parse outputs, just printed them.
        // We will assume YOLOv8 [1, 84, 8400] style or generic box.
        
        // WARNING: If this is the BoC_model from before, it might be Darknet converted.
        // If it was Darknet, it likely has [N, 85] or similar.
        // Let's implement a generic parser if possible or just the Transposed one common in v8/v5-export.
        
         // ... implementation for transposed (standard v8) above ...
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

cust::CircularList<std::string> Detector::detect(const cv::Mat& frame, std::vector<Prediction>* out_predictions) {
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
             std::cerr << "[WARN] Blob not continuous, copy might fail or be slow." << std::endl;
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
        
        std::vector<float> outputData(floatArr, floatArr + outputCount);
        
        std::vector<Prediction> predictions = _filter_predictions(outputData, outputShape, frame.size());
        
        // Sort
        _sort_bboxes(predictions);
        
        if (out_predictions) {
            *out_predictions = predictions;
        }
        
        cust::CircularList<std::string> detected_labels;
        for (const auto& pred : predictions) {
            std::string label = std::to_string(pred.classId) + " (score: " + cv::format("%.2f", pred.confidence) + ")";
            detected_labels.add(label);
        }
        
        return detected_labels;
        
    } catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] ONNX Runtime exception in detect: " << e.what() << std::endl;
        return cust::CircularList<std::string>();
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] OpenCV exception in detect: " << e.what() << std::endl;
        return cust::CircularList<std::string>();
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