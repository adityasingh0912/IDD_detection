# main.py
import os
from data_pipeline import DatasetPreparer
from train_model import train_yolov5

if __name__ == "__main__":
    base_images_dir = "JPEGImages"
    base_ann_dir = "Annotations"
    split_files = {
        "train": "train.txt",
        "val": "val.txt",
        "test": "test.txt"
    }

    # Step 1: Create dataset if not exists
    if not os.path.exists("mini_dataset"):
        preparer = DatasetPreparer(base_images_dir, base_ann_dir, split_files)
        preparer.create_mini_dataset()
    else:
        print("📂 mini_dataset already exists, skipping creation.")

    # Step 2: Train if not already trained
    if not os.path.exists("yolov5/runs/train"):
        train_yolov5("data.yaml")
    else:
        print("✅ Training already done, skipping.")
