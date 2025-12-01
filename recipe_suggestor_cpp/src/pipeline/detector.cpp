#include "data_structures/circular_list.hpp"
#include "pipeline/detector.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>

Detector::Detector(const std::string& model_path_prefix, int wid, int hei) 
    : img_width(wid), img_height(hei) {
    
    std::string cfg_path = model_path_prefix;
    std::string weights_path = model_path_prefix.substr(0, model_path_prefix.size() - 4) + ".weights";
    
    std::cout << "[INFO] Loading Darknet model from: " << cfg_path << " and " << weights_path << std::endl;
    
    net = cv::dnn::readNetFromDarknet(cfg_path, weights_path);
    
    if (net.empty()) {
        throw std::runtime_error("Failed to load model");
    }
    
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    
    std::cout << "[INFO] Model loaded successfully" << std::endl;
}

cv::Mat Detector::_preprocess_image(const cv::Mat& frame) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame, blob,
        1.0 / 255.0,
        cv::Size(img_width, img_height),
        cv::Scalar(0, 0, 0),
        true,
        false);
    
    // TODO: COVER THE REGION LATER
    // Print shapes of blob
    std::cout << "Preprocessed shape: " << blob.size << std::endl;
    return blob;
}

std::vector<Prediction> Detector::_filter_predictions(const cv::Mat& output, const cv::Size& img_size, float conf_threshold) {
    std::vector<Prediction> preds;
    
    // For classification models (output shape 1 x num_classes), treat as classification
    if (output.rows == 1 && output.cols > 1) {
        // Compute softmax probabilities
        std::vector<float> probs(output.cols);
        float max_logit = output.at<float>(0, 0);
        for (int i = 1; i < output.cols; ++i) {
            float val = output.at<float>(0, i);
            if (val > max_logit) max_logit = val;
        }
        float sum_exp = 0.0f;
        for (int i = 0; i < output.cols; ++i) {
            float logit = output.at<float>(0, i);
            float exp_val = expf(logit - max_logit);
            probs[i] = exp_val;
            sum_exp += exp_val;
        }
        for (int i = 0; i < output.cols; ++i) {
            probs[i] /= sum_exp;
        }
        // Find top 8 classes with prob > 0.01
        std::vector<std::pair<float, int>> candidates;
        for (int i = 0; i < output.cols; ++i) {
            if (probs[i] > 0.01f) {
                candidates.push_back({probs[i], i});
            }
        }
        std::sort(candidates.rbegin(), candidates.rend());
        int num = std::min(8, (int)candidates.size());
        for (int j = 0; j < num; ++j) {
            int classId = candidates[j].second;
            float conf = candidates[j].first;
            preds.push_back({ cv::Rect(0, 0, img_size.width, img_size.height), classId, conf });
        }
    } else {
        // Detection branch for YOLO
        for (int i = 0; i < output.rows; i++) {
            const float* data = output.ptr<float>(i);
            float conf = data[4];
            if (conf < conf_threshold) continue;

            int classId = std::max_element(data + 5, data + output.cols) - (data + 5);
            // Scale normalized bboxes to image size
            float x = data[0] * img_size.width;
            float y = data[1] * img_size.height;
            float w = data[2] * img_size.width;
            float h = data[3] * img_size.height;

            cv::Rect bbox(cv::Point(x - w / 2, y - h / 2), cv::Size(w, h));
            preds.push_back({ bbox, classId, conf });
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

cust::CircularList<std::string> Detector::detect(const cv::Mat& frame, std::vector<Prediction>* out_predictions) {
    try {
        // Debug network information
        std::cout << "[DEBUG] Network loaded: " << !net.empty() << std::endl;
        std::cout << "[DEBUG] Layer count: " << net.getLayerNames().size() << std::endl;
        
        // Check unconnected layers
        std::vector<int> outLayers = net.getUnconnectedOutLayers();
        std::cout << "[DEBUG] Unconnected layer indices: ";
        for (int idx : outLayers) {
            std::cout << idx << " ";
        }
        std::cout << std::endl;
        
        std::vector<cv::String> outNames = net.getUnconnectedOutLayersNames();
        std::cout << "[DEBUG] Unconnected layer names: ";
        for (const auto& name : outNames) {
            std::cout << name << " ";
        }
        std::cout << std::endl;
        
        // Preprocess
        cv::Mat input = _preprocess_image(frame);
        std::cout << "[DEBUG] Input blob - dims=" << input.dims;
        for (int i = 0; i < input.dims; ++i)
            std::cout << " size[" << i << "]=" << input.size[i];
        std::cout << " type=" << input.type() << std::endl;
        
        // Try setting input without name first
        net.setInput(input);
        std::cout << "[DEBUG] Input set successfully" << std::endl;
        
        // Try forward without specifying output names
        std::vector<cv::Mat> outputs;
        std::cout << "[DEBUG] About to call forward()..." << std::endl;
        net.forward(outputs);  // Let OpenCV figure out the outputs
        std::cout << "[DEBUG] Forward pass completed!" << std::endl;
        
        // Rest of your code...
        if (outputs.empty()) {
            std::cerr << "[ERROR] No outputs received" << std::endl;
            return cust::CircularList<std::string>();
        }
        
        std::cout << "[DEBUG] Number of outputs: " << outputs.size() << std::endl;
        for (size_t i = 0; i < outputs.size(); ++i) {
            std::cout << "[DEBUG] Output[" << i << "] - dims=" << outputs[i].dims;
            for (int j = 0; j < outputs[i].dims; ++j) {
                std::cout << " size[" << j << "]=" << outputs[i].size[j];
            }
            std::cout << std::endl;
        }
        
        // Process predictions
        std::vector<Prediction> predictions;
        for (const auto& out : outputs) {
            cv::Mat output = out;
            if (output.dims == 3) {
                output = output.reshape(1, output.size[1]);
            }
            auto preds = _filter_predictions(output, frame.size());
            predictions.insert(predictions.end(), preds.begin(), preds.end());
        }
        
        _sort_bboxes(predictions);
        
        // Add NMS
        std::vector<cv::Rect> bboxes;
        for (const auto& pred : predictions) bboxes.push_back(pred.bbox);
        std::vector<float> scores;
        for (const auto& pred : predictions) scores.push_back(pred.confidence);
        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes, scores, 0.5f, 0.4f, indices);
        
        std::vector<Prediction> filtered_predictions;
        for (int idx : indices) {
            filtered_predictions.push_back(predictions[idx]);
        }
        predictions = filtered_predictions;
        
        if (out_predictions) {
            *out_predictions = predictions;
        }
        
        cust::CircularList<std::string> detected_labels;
        for (const auto& pred : predictions) {
            std::string label = std::to_string(pred.classId) + " (score: " + cv::format("%.2f", pred.confidence) + ")";
            detected_labels.add(label);
        }
        
        return detected_labels;
        
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] OpenCV exception: " << e.what() << std::endl;
        std::cerr << "[ERROR] At line: " << e.line << " in " << e.file << std::endl;
        return cust::CircularList<std::string>();
    }
}

cv::Mat Detector::visualize_detections(const cv::Mat& frame, const std::vector<Prediction>& predictions, const std::vector<std::string>& class_names) {
    cv::Mat vis_image = frame.clone();
    for (const auto& pred : predictions) {
        cv::rectangle(vis_image, pred.bbox, cv::Scalar(0, 255, 0), 2);
        std::string class_name = (pred.classId < class_names.size()) ? class_names[pred.classId] : std::to_string(pred.classId);
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