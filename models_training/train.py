from ultralytics import YOLO
from roboflow import Roboflow
import sys
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# --- CONFIGURE ROBOFLOW SETTINGS HERE ---
# These should be set in a .env file in the same directory
ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
ROBOFLOW_WORKSPACE = os.getenv("ROBOFLOW_WORKSPACE")
ROBOFLOW_PROJECT = os.getenv("ROBOFLOW_PROJECT")
ROBOFLOW_VERSION = int(os.getenv("ROBOFLOW_VERSION", 1)) 
# ----------------------------------------

IMG_SIZE : int = 640
EPOCHS : int = 200
PATIENCE : int = 10

def download_dataset():
    """
    Downloads the dataset from Roboflow in YOLOv8 format.
    Returns the path to the data.yaml file.
    """
    print("Downloading dataset from Roboflow...")
    try:
        rf = Roboflow(api_key=ROBOFLOW_API_KEY)
        project = rf.workspace(ROBOFLOW_WORKSPACE).project(ROBOFLOW_PROJECT)
        dataset = project.version(ROBOFLOW_VERSION).download("yolov8")
        
        # The dataset.location returns the directory path. 
        # YOLOv8 expects the path to the data.yaml file.
        data_yaml_path = os.path.join(dataset.location, "data.yaml")
        
        if not os.path.exists(data_yaml_path):
             # sometimes it might be in a subdir or just 'data.yaml' in current dir if location is weird
             locations_to_check = [
                 os.path.join(dataset.location, "data.yaml"),
                 os.path.join(os.getcwd(), dataset.location, "data.yaml")
             ]
             for loc in locations_to_check:
                 if os.path.exists(loc):
                     return loc
             
             print(f"Warning: data.yaml not found at expected locations: {locations_to_check}. Returning location folder.")
             return dataset.location

        return data_yaml_path

    except Exception as e:
        print(f"Error downloading from Roboflow: {e}")
        print("Please ensure your API Key and Project details are correct.")
        sys.exit(1)

def train_and_export():
    try:
        data_path = download_dataset()
        print(f"Dataset data.yaml path: {data_path}")
        
        # Load base model
        model = YOLO('yolov8n.pt') 
        
        # Train the model
        print("Starting training...")
        results = model.train(data=data_path, epochs=EPOCHS, imgsz=IMG_SIZE, patience=PATIENCE)
        
        trained_model = YOLO('runs/detect/train/weights/best.pt') 
        
        # Test it first
        print("Testing trained model...")
        test_results = trained_model.predict('test_images/C00022_png.rf.46e3ea99a19b92bc75b7e4bd96ce5cc2.jpg', conf=0.25)
        print(f"Detections: {len(test_results[0].boxes)}")
        
        # Export the model
        print("Exporting model to ONNX...")
        success_onnx = trained_model.export(format='onnx', opset=12, simplify=True, dynamic=False)
        
        print(f"\nExport complete.")
        print(f"ONNX Model: {success_onnx}")
        
        # Copy to project
        import shutil
        shutil.copy(success_onnx, '../recipe_suggestor_cpp/resources/models/best.onnx')
        print("Copied to ../recipe_suggestor_cpp/resources/models/best.onnx")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    train_and_export()
