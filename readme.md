
# 🛣️ IDD Detection with YOLOv5

This project uses **YOLOv5** to train an object detection model on the **Indian Driving Dataset (IDD)** (mini version).  
The goal is to detect and classify various road objects from dashcam images.

---

## 📂 Project Structure

```
IDD_Detection/
│
├── yolov5/                       # YOLOv5 repository
│   ├── data/                     # Dataset config YAML files
│   ├── models/                   # Model config files
│   ├── utils/                    # Utility scripts
│   └── runs/                     # Training & evaluation outputs
│
├── mini_dataset/                 # Mini IDD dataset
│   ├── images/
    │   |── test/
│   │   ├── train/
│   │   └── val/
│   └── labels/
        |── test/
│       ├── train/
│       └── val/
│
|── data_pipeline/
|── train_model.py/
|── main.py/
|── run_inference.py/
│
└── README.md
```

---

## ⚙️ Installation

1. Clone YOLOv5 and navigate into the repo:
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
```

2. Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🚀 Training

Run the training on your dataset:

```bash
python train.py --img 640 --batch 16 --epochs 50 --data data/idd.yaml --weights yolov5s.pt
```

* `--img`: image size
* `--batch`: batch size
* `--epochs`: number of training epochs
* `--data`: dataset YAML file
* `--weights`: pre-trained weights to start from (`yolov5s.pt` for small model)

---


## 📦 Inference

Run detection on new images:

```bash
python yolov5/detect.py --weights yolov5/runs/train/exp2/weights/best.pt --source path/to/images
```

---

## 💡 Notes

* This project used the **Mini IDD dataset** for faster experiments. (A subset of IDD dataset)
* For better accuracy, use the full dataset and a larger YOLOv5 model (e.g., `yolov5m.pt`, `yolov5l.pt`).
---

## 📜 License

This project follows the [GPL-3.0 license](yolov5/LICENSE) from the YOLOv5 repository.

---
