# Models Training & Compatibility

This folder contains scripts to train a YOLOv8 object detection model and verify its compatibility with C++ (OpenCV) and Java (DJL) environments.

## 1. Prerequisites

### Python (Training)
Install the required libraries:
```bash
pip install ultralytics roboflow python-dotenv
```

### C++ (OpenCV)
Ensure you have OpenCV installed with the DNN module.

### Java (DJL)
Ensure you have Maven installed. The project relies on the same DJL dependencies defined in the main `recipe_suggestor_java/pom.xml`.

## 2. Roboflow Dataset Setup (Prerequisite)

Before running the training script, you need to set up your dataset on Roboflow:

1.  **Create a Project**: Go to Roboflow, create a new project, and select **"Object Detection"** as the project type.
2.  **Upload Data**: Upload your images and annotations.
3.  **Generate Version**:
    *   Apply any preprocessing (e.g., resizing to 640x640) or augmentations you desire.
    *   Click "Generate".
4.  **Export Settings**:
    *   When the version is generated, click **"Export Dataset"** or **"Get Snippet"**.
    *   **Format**: Select **"YOLOv8"** (this is critical: it generates the correct `data.yaml` structure).
    *   **Show Download Code**: Choose "Python".
5.  **Get Credentials**:
    *   Copy your **API Key** (kept private).
    *   Copy the **Workspace** name, **Project** name, and **Version** number from the code snippet.
6.  **Configure Environment**:
    *   Create a `.env` file in the `models_training` directory.
    *   Add the following variables:
        ```env
        ROBOFLOW_API_KEY=your_private_api_key
        ROBOFLOW_WORKSPACE=your_workspace_name
        ROBOFLOW_PROJECT=your_project_name
        ROBOFLOW_VERSION=1
        ```

## 3. Training the Model

1.  Ensure you have created the `.env` file with your Roboflow credentials.
2.  Run the script:
    ```bash
    python3 train.py
    ```
3.  The script will:
    *   Download your dataset from Roboflow.
    *   Train a YOLOv8 Nano model for 100 epochs.
    *   Export the best model to `best.onnx` and `best.torchscript`.

## 4. Verifying C++ Compatibility

Run the setup script to download ONNX Runtime (required for modern YOLOv8 support):
```bash
./setup_cpp_env.sh
```

Compile `test_cpp.cpp`. We link against both OpenCV (for image loading) and ONNX Runtime (for inference).
Adjust paths if your ONNX Runtime version differs from `1.16.3`.

```bash
g++ test_cpp.cpp -o test_cpp \
    -I lib/onnxruntime-linux-x64-1.16.3/include \
    -L lib/onnxruntime-linux-x64-1.16.3/lib \
    -lonnxruntime \
    `pkg-config --cflags --libs opencv4` \
    -Wl,-rpath,lib/onnxruntime-linux-x64-1.16.3/lib
```

Run the executable:
```bash
./test_cpp best.onnx
```

Expected output should show "Model loaded successfully!" and the tensor shape of the output.

## 5. Verifying Java Compatibility

This folder contains a dedicated Maven project to test if the exported ONNX model can be loaded and run using the Java DJL library (which is used in the main application).

1.  Navigate to the test project directory:
    ```bash
    cd java_compatibility_test
    ```
2.  Run the test using Maven. You need to provide the path to the `best.onnx` file (which is in the parent directory):
    ```bash
    mvn clean compile exec:java -Dexec.args="../best.onnx"
    ```

3.  Expected output:
    ```text
    Attempting to load model from: ../best.onnx
    ...
    Model loaded successfully!
    Running inference on dummy 1x3x640x640 input...
    Inference successful.
    Output shape: ...
    ```