# train_model.py
import os
import subprocess

def train_yolov5(data_yaml, weights="yolov5s.pt", img_size=640, batch=8, epochs=5):
    yolov5_train = os.path.join("yolov5", "train.py")
    cmd = [
        "python", yolov5_train,
        "--data", data_yaml,
        "--weights", weights,
        "--img", str(img_size),
        "--batch", str(batch),
        "--epochs", str(epochs)
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    train_yolov5("data.yaml")
