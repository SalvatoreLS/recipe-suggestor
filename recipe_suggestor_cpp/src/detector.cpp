#include "data_structures/circular_list.hpp"
#include "types.hpp"
#include "detector.hpp"
#include "utils.hpp"
#include <iostream>
#include <cmath>
#include <algorithm>
#include <numeric>

Detector::Detector(const std::string& model_path, int wid, int hei, types::Preprocess preprocess)
    : img_width(wid), img_height(hei), preprocess_(preprocess),
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

        load_class_names();

        // The output is [1, 4 + num_classes, anchors]; if that does not line up
        // with the names the model reports, something is badly out of sync.
        if (numOutputNodes > 0 && !class_names_.empty()) {
            auto shape = session.GetOutputTypeInfo(0).GetTensorTypeAndShapeInfo().GetShape();
            if (shape.size() == 3 && shape[1] > 0 &&
                shape[1] != static_cast<int64_t>(4 + class_names_.size())) {
                throw std::runtime_error(
                    "Model output channel count (" + std::to_string(shape[1]) +
                    ") does not match 4 + " + std::to_string(class_names_.size()) + " classes");
            }
        }

    } catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] ONNX Runtime exception: " << e.what() << "\n";
        throw;
    }
}

// Ultralytics stores the class list in the ONNX metadata as a Python-repr
// dict: {0: 'black_heart', 1: 'bomb', ...}. Scanning it out beats shipping a
// separate synset.txt that can silently belong to the wrong model.
void Detector::load_class_names() {
    std::string raw;
    try {
        auto meta = session.GetModelMetadata();
        auto value = meta.LookupCustomMetadataMapAllocated("names", allocator);
        if (!value) return;
        raw = value.get();
    } catch (const Ort::Exception&) {
        return;
    }

    std::map<int, std::string> by_index;
    size_t i = 0;
    while (i < raw.size()) {
        // index
        while (i < raw.size() && !isdigit(static_cast<unsigned char>(raw[i]))) i++;
        if (i >= raw.size()) break;
        size_t num_start = i;
        while (i < raw.size() && isdigit(static_cast<unsigned char>(raw[i]))) i++;
        int index = std::stoi(raw.substr(num_start, i - num_start));

        // colon, then a quoted name
        while (i < raw.size() && raw[i] != ':' && raw[i] != '\'' && raw[i] != '"') i++;
        if (i >= raw.size() || raw[i] != ':') continue;
        i++;
        while (i < raw.size() && raw[i] != '\'' && raw[i] != '"') i++;
        if (i >= raw.size()) break;
        char quote = raw[i++];
        size_t name_start = i;
        while (i < raw.size() && raw[i] != quote) i++;
        if (i >= raw.size()) break;
        by_index[index] = raw.substr(name_start, i - name_start);
        i++;
    }

    if (by_index.empty()) return;
    class_names_.resize(by_index.rbegin()->first + 1);
    for (const auto& [idx, name] : by_index) class_names_[idx] = name;
}

cv::Mat Detector::_preprocess_image(const cv::Mat& frame) {
    // Geometry must match how THIS model's dataset was built -- the two
    // datasets differ, so the mode is a per-detector property. See
    // types::Preprocess.
    cv::Mat resized;
    if (preprocess_ == types::Preprocess::LetterboxBlack) {
        float scale = 1.0f;
        resized = letterbox(frame, cv::Size(img_width, img_height), scale,
                            pad_x_, pad_y_, cv::Scalar(0, 0, 0));
        scale_x_ = scale_y_ = scale;
    } else {
        // Squash to the model size exactly as Roboflow squashed the training
        // images: independent axis scales, no padding.
        cv::resize(frame, resized, cv::Size(img_width, img_height));
        scale_x_ = static_cast<float>(img_width) / frame.cols;
        scale_y_ = static_cast<float>(img_height) / frame.rows;
        pad_x_ = pad_y_ = 0;
    }

    cv::Mat blob;
    cv::dnn::blobFromImage(
        resized,                        // already at model size
        blob,
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

                 // Undo the letterbox: strip the padding, then the scale.
                 int x = (int)((cx - 0.5f * w - pad_x_) / scale_x_);
                 int y = (int)((cy - 0.5f * h - pad_y_) / scale_y_);
                 int width = (int)(w / scale_x_);
                 int height = (int)(h / scale_y_);

                 cv::Rect box = cv::Rect(x, y, width, height) & cv::Rect(0, 0, img_size.width, img_size.height);
                 if (box.width <= 0 || box.height <= 0) continue;

                 bboxes.push_back(box);
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

std::vector<Prediction> Detector::run_inference(const cv::Mat& frame, float conf_threshold) {
    try {
        // 1. Preprocess
        cv::Mat blob = _preprocess_image(frame);

        // 2. Prepare Input Tensor
        size_t inputTensorSize = 1 * 3 * img_width * img_height;
        std::vector<float> inputTensorValues(inputTensorSize);

        if (blob.isContinuous()) {
            memcpy(inputTensorValues.data(), blob.ptr<float>(), inputTensorSize * sizeof(float));
        } else {
             cv::Mat cont = blob.clone();
             memcpy(inputTensorValues.data(), cont.ptr<float>(), inputTensorSize * sizeof(float));
        }

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
        std::cout << "Output shape: [";
        for (size_t i = 0; i < outputShape.size(); i++) {
            std::cout << outputShape[i];
            if (i < outputShape.size() - 1) std::cout << ", ";
        }
        std::cout << "], elements: " << outputCount << std::endl;
        #endif

        std::vector<float> outputData(floatArr, floatArr + outputCount);
        return _filter_predictions(outputData, outputShape, frame.size(), conf_threshold);

    } catch (const Ort::Exception& e) {
        std::cerr << "[ERROR] ONNX Runtime exception in run_inference: " << e.what() << std::endl;
        return {};
    } catch (const cv::Exception& e) {
        std::cerr << "[ERROR] OpenCV exception in run_inference: " << e.what() << std::endl;
        return {};
    }
}

cust::CircularList<types::ConsumableID> Detector::detect(const cv::Mat& frame, std::vector<Prediction>* out_predictions) {
    std::vector<Prediction> predictions = run_inference(frame);

    // Reading order, so the bag comes out left-to-right.
    _sort_bboxes(predictions);

    if (out_predictions) *out_predictions = predictions;

    cust::CircularList<types::ConsumableID> detected_items;
    for (const auto& pred : predictions) {
        detected_items.add(static_cast<types::ConsumableID>(pred.classId));
    }

    return detected_items;
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
