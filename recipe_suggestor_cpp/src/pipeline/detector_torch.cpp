#include "data_structures/circular_list.hpp"
#include "pipeline/detector.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <torch/torch.h>
#include <torch/script.h>
#include <fstream>
#include <filesystem>


std::vector<std::string> load_class_names(const std::string& path) {
    std::vector<std::string> names;
    std::ifstream in(path);
    if (!in.is_open()) {
        throw std::runtime_error("Cannot open synset file: " + path);
    }
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        size_t pos = line.find('|');
        if (pos != std::string::npos) {
            std::string name = line.substr(pos + 2); // skip | and space
            names.push_back(name);
        } else {
            names.push_back(line);
        }
    }
    return names;
}

Detector::Detector(const std::string& model_path_prefix, int wid, int hei) 
    : img_width(wid), img_height(hei) {
    
    std::string model_path = model_path_prefix;
    
    std::cout << "[INFO] Loading TorchScript model from: " << model_path << std::endl;
    
    try {
        model = torch::jit::load(model_path);
        model.eval();
        
        std::filesystem::path model_path_obj(model_path_prefix);
        std::filesystem::path synset_path = std::filesystem::absolute(model_path_obj).parent_path() / "synset.txt";
        class_names = load_class_names(synset_path.string());
        std::cout << "[INFO] Loaded " << class_names.size() << " class names" << std::endl;
    } catch (const c10::Error& e) {
        throw std::runtime_error(std::string("Failed to load model: ") + std::string(e.what()));
    }
}

cv::Mat Detector::_preprocess_image(const cv::Mat& frame) {
    cv::Mat blob;
    cv::dnn::blobFromImage(frame,
        blob,
        1.0 / 255.0,
        cv::Size(img_width, img_height),
        cv::Scalar(0, 0, 0),
        true,
        false);
    return blob;
}

torch::Tensor Detector::_mat_to_tensor(const cv::Mat& mat) {
    // Convert cv::Mat to torch::Tensor
    torch::Tensor tensor = torch::from_blob(mat.data, {1, 3, img_height, img_width}, torch::kFloat32);
    return tensor;
}

std::vector<Prediction> Detector::_filter_predictions(const torch::Tensor& output, const cv::Size& img_size, float conf_threshold) {
    std::vector<Prediction> preds;
    
    // YOLO output parsing
    auto output_cpu = output.to(torch::kCPU);
    auto sizes = output_cpu.sizes();
    int num_detections = sizes[1];
    int num_features = sizes[2];
    int num_classes = num_features - 4;  // 4 for x,y,w,h, then class logits
    
    for (int i = 0; i < num_detections; ++i) {
        // Class logits
        std::vector<float> class_logits(num_classes);
        for (int c = 0; c < num_classes; ++c) {
            class_logits[c] = output_cpu[0][i][4 + c].item<float>();
        }

        // Softmax to get probabilities
        std::vector<float> class_probs(num_classes);
        float max_logit = *std::max_element(class_logits.begin(), class_logits.end());
        float sum = 0.0f;
        for (float logit : class_logits) {
            sum += exp(logit - max_logit);  // for numerical stability
        }
        for (int c = 0; c < num_classes; ++c) {
            class_probs[c] = exp(class_logits[c] - max_logit) / sum;
        }
        
        float max_class_prob = *std::max_element(class_probs.begin(), class_probs.end());
        int classId = std::max_element(class_probs.begin(), class_probs.end()) - class_probs.begin();
        
        float conf = max_class_prob;
        if (conf < conf_threshold) continue;
        
        float x = output_cpu[0][i][0].item<float>() * img_size.width / img_width;
        float y = output_cpu[0][i][1].item<float>() * img_size.height / img_height;
        float w = output_cpu[0][i][2].item<float>() * img_size.width / img_width;
        float h = output_cpu[0][i][3].item<float>() * img_size.height / img_height;
        
        float x_min = std::max(0.0f, x - w / 2);
        float y_min = std::max(0.0f, y - h / 2);
        w = std::min(w, img_size.width - x_min);
        h = std::min(h, img_size.height - y_min);
        
        cv::Rect bbox(cv::Point(x_min, y_min), cv::Size(w, h));
        preds.push_back({ bbox, classId, conf });
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
        // Preprocess
        cv::Mat input_mat = _preprocess_image(frame);
        torch::Tensor input_tensor = _mat_to_tensor(input_mat);
        
        // Forward pass
        std::vector<torch::jit::IValue> inputs;
        inputs.push_back(input_tensor);
        auto output = model.forward(inputs).toTensor();
        
        // Check if output needs transpose [1, 25, 8400] -> [1, 8400, 25]
        if (output.size(1) < output.size(2)) {
            output = output.transpose(1, 2);
        }

        // Process predictions
        std::vector<Prediction> predictions = _filter_predictions(output, frame.size());
        
        _sort_bboxes(predictions);
        
        // Add NMS
        std::vector<cv::Rect> bboxes;
        for (const auto& pred : predictions) bboxes.push_back(pred.bbox);
        std::vector<float> scores;
        for (const auto& pred : predictions) scores.push_back(pred.confidence);
        std::vector<int> indices;
        cv::dnn::NMSBoxes(bboxes, scores, 0.1f, 0.4f, indices);
        
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
            std::string label = class_names[pred.classId] + " (score: " + cv::format("%.2f", pred.confidence) + ")";
            label += "   Bounding box: x=" + std::to_string(pred.bbox.x) + ", y=" + std::to_string(pred.bbox.y) +
                      ", w=" + std::to_string(pred.bbox.width) + ", h=" + std::to_string(pred.bbox.height) + " ";
            detected_labels.add(label);
        }
        
        return detected_labels;
        
    } catch (const c10::Error& e) {
        std::cerr << "[ERROR] Torch exception: " << e.what() << std::endl;
        return cust::CircularList<std::string>();
    }
}

cv::Mat Detector::visualize_detections(const cv::Mat& frame, const std::vector<Prediction>& predictions, const std::vector<std::string>& class_names) {
    cv::Mat vis_image = frame.clone();
    for (const auto& pred : predictions) {
        cv::Rect bbox = pred.bbox & cv::Rect(0, 0, frame.cols, frame.rows);
        cv::rectangle(vis_image, bbox, cv::Scalar(0, 255, 0), 2);
        std::string class_name = (pred.classId < (int)class_names.size()) ? class_names[pred.classId] : std::to_string(pred.classId);
        std::string label = class_name + ": " + cv::format("%.2f", pred.confidence);
        int baseLine;
        cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
        int top = std::max(bbox.y, labelSize.height);
        cv::rectangle(vis_image, cv::Point(bbox.x, top - labelSize.height),
                      cv::Point(bbox.x + labelSize.width, top + baseLine),
                      cv::Scalar(255, 255, 255), cv::FILLED);
        cv::putText(vis_image, label, cv::Point(bbox.x, top),
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0), 1);
    }
    return vis_image;
}
