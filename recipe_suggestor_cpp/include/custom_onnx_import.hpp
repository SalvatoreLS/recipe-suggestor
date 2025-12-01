#ifndef CUST_DNN_ONNX_READER_HPP
#define CUST_DNN_ONNX_READER_HPP

#include <opencv2/dnn.hpp>
#include <string>
#include <vector>
#include <unordered_map>
#include <fstream>
#include <cstdint>

namespace cust {
namespace dnn {

/**
 * @brief Read an ONNX model and return an OpenCV DNN network
 * @param onnx_filename Path to the ONNX file
 * @return OpenCV DNN Net object
 */
cv::dnn::Net readONNX(const std::string& onnx_filename);

namespace internal {

// ONNX data structures for parsing
struct ONNXTensor {
    std::string name;
    std::vector<int64_t> dims;
    std::vector<float> float_data;
    std::vector<int64_t> int64_data;
    int data_type; // 1=float, 7=int64, etc.
};

struct ONNXAttribute {
    std::string name;
    int type; // 1=float, 2=int, 3=string, 6=floats, 7=ints
    float f;
    int64_t i;
    std::string s;
    std::vector<float> floats;
    std::vector<int64_t> ints;
};

struct ONNXNode {
    std::string op_type;
    std::string name;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::vector<ONNXAttribute> attributes;
};

struct ONNXGraph {
    std::vector<ONNXNode> nodes;
    std::vector<ONNXTensor> initializers;
    std::vector<std::string> inputs;
    std::vector<std::string> outputs;
    std::string name;
};

// Helper functions
uint64_t readVarintFromBuffer(const std::vector<uint8_t>& buffer, size_t& offset);
std::string readStringFromBuffer(const std::vector<uint8_t>& buffer, size_t& offset, size_t length);
ONNXAttribute parseAttribute(const std::vector<uint8_t>& attrData);
ONNXTensor parseTensor(const std::vector<uint8_t>& tensorData);
ONNXNode parseNode(const std::vector<uint8_t>& nodeData);
std::string parseValueInfo(const std::vector<uint8_t>& valueInfoData);
ONNXGraph parseGraph(const std::vector<uint8_t>& graphData);
std::vector<uint8_t> readFile(const std::string& filename);

// Convert ONNX to OpenCV
cv::dnn::Net buildOpenCVNet(const ONNXGraph& graph);
cv::dnn::LayerParams createLayerParams(const ONNXNode& node, 
                                       const std::unordered_map<std::string, ONNXTensor>& weights_map);

} // namespace internal
} // namespace dnn
} // namespace cust

#endif // CUST_DNN_ONNX_READER_HPP