from ultralytics import YOLO
import os
from pathlib import Path
from tqdm import tqdm

TEST_IMG_PATH : Path = Path('test_images')
OUTPUT_PATH : Path = Path('outputs')
MODEL_PATH : Path = Path('runs/detect/train/weights/best.pt')

def predict_and_save(model : YOLO):
    for img in tqdm(os.listdir(path=TEST_IMG_PATH), desc="Testing model..."):
        results = model(TEST_IMG_PATH / img)
        for result in results: result.save(filename=OUTPUT_PATH / img)

def main():
    model = YOLO(MODEL_PATH)
    predict_and_save(model=model)


if __name__ == "__main__":
    main()