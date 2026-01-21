#!/bin/bash

# Configuration
ORT_VERSION="1.16.3"
ORT_OS="linux"
ORT_ARCH="x64"
ORT_NAME="onnxruntime-${ORT_OS}-${ORT_ARCH}-${ORT_VERSION}"
ORT_URL="https://github.com/microsoft/onnxruntime/releases/download/v${ORT_VERSION}/${ORT_NAME}.tgz"
INSTALL_DIR="lib"

# Create lib directory
mkdir -p ${INSTALL_DIR}

# Download and extract if not already present
if [ ! -d "${INSTALL_DIR}/${ORT_NAME}" ]; then
    echo "Downloading ONNX Runtime v${ORT_VERSION}..."
    wget -q --show-progress ${ORT_URL} -O ${INSTALL_DIR}/ort.tgz
    
    echo "Extracting..."
    tar -xzf ${INSTALL_DIR}/ort.tgz -C ${INSTALL_DIR}
    
    echo "Cleaning up..."
    rm ${INSTALL_DIR}/ort.tgz
    
    echo "ONNX Runtime installed to ${INSTALL_DIR}/${ORT_NAME}"
else
    echo "ONNX Runtime already installed at ${INSTALL_DIR}/${ORT_NAME}"
fi
