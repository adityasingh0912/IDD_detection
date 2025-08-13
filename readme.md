
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

## 📊 Results & Outputs

After training, results are stored in:

```
yolov5/runs/train/expX/
```

Key files:

* `results.png` → Training curves (loss, mAP, precision, recall)
* `confusion_matrix.png` → Confusion matrix of predictions
* `PR_curve.png` → Precision-Recall curves per class
* `labels.jpg` → Distribution of bounding boxes in the dataset
* `best.pt` → Best model weights (highest mAP)
* `last.pt` → Last epoch weights

Example training curve (`results.png`):
![results](yolov5/runs/train/exp2/results.png)

Example confusion matrix (`confusion_matrix.png`):
![confusion matrix](yolov5/runs/train/exp2/confusion_matrix.png)

---

## 🧪 Evaluation

To evaluate your trained model and generate a confusion matrix:

```bash
python confusion_matrix.py --weights yolov5/runs/train/exp2/weights/best.pt --data yolov5/data/idd.yaml --device 0
```

> **Tip:** If you don't have a GPU, this will run slowly on CPU.
> For faster evaluation, use a cloud GPU (Google Colab / Kaggle).

---

## 📦 Inference

Run detection on new images:

```bash
python yolov5/detect.py --weights yolov5/runs/train/exp2/weights/best.pt --source path/to/images
```

---

## 💡 Notes

* This project used the **Mini IDD dataset** for faster experiments.
* For better accuracy, use the full dataset and a larger YOLOv5 model (e.g., `yolov5m.pt`, `yolov5l.pt`).
* The confusion matrix in `yolov5/runs/train/exp2/confusion_matrix.png` is already generated at the end of training — no need to re-run unless testing new weights/dataset.

---

## 📜 License

This project follows the [GPL-3.0 license](yolov5/LICENSE) from the YOLOv5 repository.

---
