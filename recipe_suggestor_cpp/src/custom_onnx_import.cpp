#include "custom_onnx_import.hpp"
#include <opencv2/dnn.hpp>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <cstring>
#include <iostream>

namespace cust {
namespace dnn {
namespace internal {

uint64_t readVarintFromBuffer(const std::vector<uint8_t>& buffer, size_t& offset) {
    uint64_t result = 0;
    int shift = 0;
    uint8_t byte;
    
    do {
        if (offset >= buffer.size()) {
            throw std::runtime_error("Buffer overflow while reading varint");
        }
        byte = buffer[offset++];
        result |= static_cast<uint64_t>(byte & 0x7F) << shift;
        shift += 7;
    } while (byte & 0x80);
    
    return result;
}

std::string readStringFromBuffer(const std::vector<uint8_t>& buffer, size_t& offset, size_t length) {
    if (offset + length > buffer.size()) {
        throw std::runtime_error("Buffer overflow while reading string");
    }
    std::string result(buffer.begin() + offset, buffer.begin() + offset + length);
    offset += length;
    return result;
}

ONNXAttribute parseAttribute(const std::vector<uint8_t>& attrData) {
    ONNXAttribute attr;
    attr.type = 0; // Initialize type
    size_t offset = 0;
    
    while (offset < attrData.size()) {
        if (offset + 1 >= attrData.size()) break;
        
        uint8_t tag = attrData[offset++];
        uint8_t wireType = tag & 0x07;
        uint8_t fieldNumber = tag >> 3;
        
        if (wireType == 2) { // Length-delimited
            uint64_t length = readVarintFromBuffer(attrData, offset);
            
            if (offset + length > attrData.size()) break;
            
            if (fieldNumber == 1) { // name
                attr.name = readStringFromBuffer(attrData, offset, length);
            } else if (fieldNumber == 3) { // string value (s)
                attr.s = readStringFromBuffer(attrData, offset, length);
            } else if (fieldNumber == 6) { // floats array
                // Read raw bytes and interpret as floats
                size_t float_count = length / sizeof(float);
                attr.floats.resize(float_count);
                if (offset + length <= attrData.size()) {
                    std::memcpy(attr.floats.data(), &attrData[offset], length);
                }
                offset += length;
            } else if (fieldNumber == 7) { // ints array (packed repeated)
                size_t end = offset + length;
                while (offset < end && offset < attrData.size()) {
                    attr.ints.push_back(static_cast<int64_t>(readVarintFromBuffer(attrData, offset)));
                }
            } else {
                offset += length;
            }
        } else if (wireType == 0) { // Varint
            uint64_t value = readVarintFromBuffer(attrData, offset);
            
            if (fieldNumber == 20) { // type (THIS IS THE KEY FIX)
                attr.type = static_cast<int>(value);
            } else if (fieldNumber == 2) { // float value (f)
                // This is tricky - ONNX uses varint wire type for the field tag,
                // but the actual float is 32-bit fixed
                // We need to handle this differently
            } else if (fieldNumber == 4) { // int value (i)
                attr.i = value;
            }
        } else if (wireType == 5) { // 32-bit fixed (for float)
            if (offset + 4 <= attrData.size()) {
                if (fieldNumber == 2) { // float value (f)
                    std::memcpy(&attr.f, &attrData[offset], sizeof(float));
                }
                offset += 4;
            }
        } else if (wireType == 1) { // 64-bit fixed
            if (offset + 8 <= attrData.size()) {
                // Skip for now
                offset += 8;
            }
        }
    }
    
    return attr;
}

ONNXTensor parseTensor(const std::vector<uint8_t>& tensorData) {
    ONNXTensor tensor;
    size_t offset = 0;
    
    while (offset < tensorData.size()) {
        if (offset + 1 >= tensorData.size()) break;
        
        uint8_t tag = tensorData[offset++];
        uint8_t wireType = tag & 0x07;
        uint8_t fieldNumber = tag >> 3;
        
        if (wireType == 2) { // Length-delimited
            uint64_t length = readVarintFromBuffer(tensorData, offset);
            
            if (fieldNumber == 1) { // name
                tensor.name = readStringFromBuffer(tensorData, offset, length);
            } else if (fieldNumber == 2) { // dims
                size_t end = offset + length;
                while (offset < end && offset < tensorData.size()) {
                    tensor.dims.push_back(readVarintFromBuffer(tensorData, offset));
                }
            } else if (fieldNumber == 5) { // float_data
                size_t count = length / sizeof(float);
                tensor.float_data.resize(count);
                if (offset + length <= tensorData.size()) {
                    std::memcpy(tensor.float_data.data(), &tensorData[offset], length);
                }
                offset += length;
            } else if (fieldNumber == 9) { // raw_data
                size_t count = length / sizeof(float);
                tensor.float_data.resize(count);
                if (offset + length <= tensorData.size()) {
                    std::memcpy(tensor.float_data.data(), &tensorData[offset], length);
                }
                offset += length;
            } else {
                offset += length;
            }
        } else if (wireType == 0) { // Varint
            uint64_t value = readVarintFromBuffer(tensorData, offset);
            if (fieldNumber == 3) { // data_type
                tensor.data_type = value;
            }
        }
    }
    
    return tensor;
}

ONNXNode parseNode(const std::vector<uint8_t>& nodeData) {
    ONNXNode node;
    size_t offset = 0;
    
    while (offset < nodeData.size()) {
        if (offset + 1 >= nodeData.size()) break;
        
        uint8_t tag = nodeData[offset++];
        uint8_t wireType = tag & 0x07;
        uint8_t fieldNumber = tag >> 3;
        
        if (wireType == 2) { // Length-delimited
            uint64_t length = readVarintFromBuffer(nodeData, offset);
            
            if (offset + length > nodeData.size()) break;
            
            if (fieldNumber == 1) { // inputs
                node.inputs.push_back(readStringFromBuffer(nodeData, offset, length));
            } else if (fieldNumber == 2) { // outputs
                node.outputs.push_back(readStringFromBuffer(nodeData, offset, length));
            } else if (fieldNumber == 3) { // name
                node.name = readStringFromBuffer(nodeData, offset, length);
            } else if (fieldNumber == 4) { // op_type
                node.op_type = readStringFromBuffer(nodeData, offset, length);
            } else if (fieldNumber == 5) { // attributes
                std::vector<uint8_t> attrData(nodeData.begin() + offset,
                                              nodeData.begin() + offset + length);
                node.attributes.push_back(parseAttribute(attrData));
                offset += length;
            } else {
                offset += length;
            }
        } else if (wireType == 0) {
            readVarintFromBuffer(nodeData, offset);
        }
    }
    
    return node;
}

std::string parseValueInfo(const std::vector<uint8_t>& valueInfoData) {
    std::string name;
    size_t offset = 0;
    
    while (offset < valueInfoData.size()) {
        if (offset + 1 >= valueInfoData.size()) break;
        
        uint8_t tag = valueInfoData[offset++];
        uint8_t wireType = tag & 0x07;
        uint8_t fieldNumber = tag >> 3;
        
        if (wireType == 2) {
            uint64_t length = readVarintFromBuffer(valueInfoData, offset);
            
            if (fieldNumber == 1) { // name
                name = readStringFromBuffer(valueInfoData, offset, length);
            } else {
                offset += length;
            }
        } else if (wireType == 0) {
            readVarintFromBuffer(valueInfoData, offset);
        }
    }
    
    return name;
}

ONNXGraph parseGraph(const std::vector<uint8_t>& graphData) {
    ONNXGraph graph;
    size_t offset = 0;
    
    while (offset < graphData.size()) {
        if (offset + 1 >= graphData.size()) break;
        
        uint8_t tag = graphData[offset++];
        uint8_t wireType = tag & 0x07;
        uint8_t fieldNumber = tag >> 3;
        
        if (wireType == 2) {
            uint64_t length = readVarintFromBuffer(graphData, offset);
            
            if (offset + length > graphData.size()) break;
            
            std::vector<uint8_t> fieldData(graphData.begin() + offset,
                                          graphData.begin() + offset + length);
            
            if (fieldNumber == 1) { // nodes
                graph.nodes.push_back(parseNode(fieldData));
            } else if (fieldNumber == 2) { // name
                graph.name = readStringFromBuffer(graphData, offset, length);
                continue; // already consumed
            } else if (fieldNumber == 5) { // initializers
                graph.initializers.push_back(parseTensor(fieldData));
            } else if (fieldNumber == 11) { // inputs
                std::string input_name = parseValueInfo(fieldData);
                if (!input_name.empty()) {
                    graph.inputs.push_back(input_name);
                }
            } else if (fieldNumber == 12) { // outputs
                std::string output_name = parseValueInfo(fieldData);
                if (!output_name.empty()) {
                    graph.outputs.push_back(output_name);
                }
            }
            
            offset += length;
        } else if (wireType == 0) {
            readVarintFromBuffer(graphData, offset);
        }
    }
    
    return graph;
}

std::vector<uint8_t> readFile(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open ONNX file: " + filename);
    }
    
    file.seekg(0, std::ios::end);
    size_t fileSize = file.tellg();
    file.seekg(0, std::ios::beg);
    
    if (fileSize < 4) {
        throw std::runtime_error("Invalid ONNX file: too small");
    }
    
    std::vector<uint8_t> buffer(fileSize);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();
    
    return buffer;
}

cv::dnn::LayerParams createLayerParams(const ONNXNode& node,
                                       const std::unordered_map<std::string, ONNXTensor>& weights_map) {
    cv::dnn::LayerParams params;
    params.name = node.name;
    params.type = node.op_type;
    
    std::cout << "Creating layer: " << node.name << " (type: " << node.op_type << ")" << std::endl;
    
    // Parse attributes based on layer type
    for (const auto& attr : node.attributes) {
        std::cout << "  Attribute: " << attr.name << " (type: " << attr.type << ")" << std::endl;
        
        if (attr.type == 1) { // float
            std::cout << "    Float value: " << attr.f << std::endl;
            params.set(attr.name, attr.f);
        } else if (attr.type == 2) { // int
            std::cout << "    Int value: " << attr.i << std::endl;
            params.set(attr.name, static_cast<int>(attr.i));
        } else if (attr.type == 3) { // string
            std::cout << "    String value: " << attr.s << std::endl;
            params.set(attr.name, attr.s);
        } else if (attr.type == 6) { // floats array
            if (!attr.floats.empty()) {
                std::cout << "    Float array size: " << attr.floats.size() << std::endl;
                cv::dnn::DictValue dv = cv::dnn::DictValue::arrayReal(attr.floats.data(), attr.floats.size());
                params.set(attr.name, dv);
            }
        } else if (attr.type == 7) { // ints array
            if (!attr.ints.empty()) {
                std::cout << "    Int array size: " << attr.ints.size() << std::endl;
                std::vector<int> int_vals;
                for (auto val : attr.ints) {
                    int_vals.push_back(static_cast<int>(val));
                }
                cv::dnn::DictValue dv = cv::dnn::DictValue::arrayInt(int_vals.data(), int_vals.size());
                params.set(attr.name, dv);
            }
        } else {
            std::cout << "    Unknown attribute type: " << attr.type << std::endl;
        }
    }
    
    // Add weights/biases if they exist
    for (size_t i = 1; i < node.inputs.size(); ++i) {
        auto it = weights_map.find(node.inputs[i]);
        if (it != weights_map.end()) {
            const ONNXTensor& tensor = it->second;
            
            // Convert dims to cv::Mat shape
            std::vector<int> shape;
            for (auto dim : tensor.dims) {
                shape.push_back(static_cast<int>(dim));
            }
            
            // Create cv::Mat from tensor data
            cv::Mat blob;
            if (!tensor.float_data.empty() && !shape.empty()) {
                blob = cv::Mat(shape, CV_32F, const_cast<float*>(tensor.float_data.data())).clone();
                params.blobs.push_back(blob);
                std::cout << "  Added blob with shape: ";
                for (int s : shape) std::cout << s << " ";
                std::cout << std::endl;
            }
        }
    }
    
    return params;
}

cv::dnn::Net buildOpenCVNet(const ONNXGraph& graph) {
    cv::dnn::Net net;
    
    // Create a map of initializers (weights) for quick lookup
    std::unordered_map<std::string, ONNXTensor> weights_map;
    for (const auto& tensor : graph.initializers) {
        weights_map[tensor.name] = tensor;
    }
    
    // Track tensor names to layer IDs
    std::unordered_map<std::string, int> tensor_to_layer;
    
    // Add input layer
    if (!graph.inputs.empty()) {
        cv::dnn::LayerParams inputParams;
        inputParams.name = graph.inputs[0];
        inputParams.type = "Input";
        int inputLayerId = net.addLayer(inputParams.name, inputParams.type, inputParams);
        tensor_to_layer[graph.inputs[0]] = inputLayerId;
    }
    
    // Add each node as a layer
    for (const auto& node : graph.nodes) {
        cv::dnn::LayerParams params = createLayerParams(node, weights_map);
        
        // Add layer to network
        int layerId = net.addLayer(params.name, params.type, params);
        
        // Connect inputs
        std::vector<int> inputLayers;
        for (const auto& input : node.inputs) {
            auto it = tensor_to_layer.find(input);
            if (it != tensor_to_layer.end()) {
                inputLayers.push_back(it->second);
            }
        }
        
        if (!inputLayers.empty()) {
            net.connect(inputLayers[0], 0, layerId, 0);
        }
        
        // Map outputs to this layer
        for (const auto& output : node.outputs) {
            tensor_to_layer[output] = layerId;
        }
    }
    
    return net;
}

} // namespace internal

cv::dnn::Net readONNX(const std::string& onnx_filename) {
    // Read the ONNX file
    std::vector<uint8_t> buffer = internal::readFile(onnx_filename);
    
    // Parse the protobuf structure to find the graph
    size_t offset = 0;
    internal::ONNXGraph graph;
    
    while (offset < buffer.size()) {
        if (offset + 1 >= buffer.size()) break;
        
        uint8_t tag = buffer[offset++];
        uint8_t wireType = tag & 0x07;
        uint8_t fieldNumber = tag >> 3;
        
        if (wireType == 2) {
            uint64_t length = internal::readVarintFromBuffer(buffer, offset);
            
            if (offset + length > buffer.size()) break;
            
            // Field 7 in ModelProto is the graph
            if (fieldNumber == 7) {
                std::vector<uint8_t> graphData(buffer.begin() + offset,
                                               buffer.begin() + offset + length);
                graph = internal::parseGraph(graphData);
            }
            
            offset += length;
        } else if (wireType == 0) {
            internal::readVarintFromBuffer(buffer, offset);
        } else {
            break;
        }
    }
    
    // Build and return the OpenCV network
    return internal::buildOpenCVNet(graph);
}

} // namespace dnn
} // namespace cust