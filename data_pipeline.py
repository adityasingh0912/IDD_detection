import os
import random
import shutil
import math
import xml.etree.ElementTree as ET

# Paths
base_images_dir = "JPEGImages"
base_ann_dir = "Annotations"
mini_base = "mini_dataset"

splits = {
    "train": "train.txt",
    "val": "val.txt",
    "test": "test.txt"
}

# Get all class names from XML files recursively
def get_classes():
    classes_set = set()
    for root_dir, _, files in os.walk(base_ann_dir):
        for xml_file in files:
            if not xml_file.lower().endswith(".xml"):
                continue
            xml_path = os.path.join(root_dir, xml_file)
            try:
                tree = ET.parse(xml_path)
                root = tree.getroot()
                for obj in root.findall("object"):
                    classes_set.add(obj.find("name").text)
            except Exception as e:
                print(f"⚠ Error reading {xml_path}: {e}")
    return sorted(list(classes_set))

classes = get_classes()
print(f"Detected classes: {classes}")

# Convert PASCAL XML to YOLO format
def convert_xml_to_yolo(xml_path, yolo_txt_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)

    lines = []
    for obj in root.findall("object"):
        cls_name = obj.find("name").text
        if cls_name not in classes:
            continue
        cls_id = classes.index(cls_name)
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # YOLO format
        x_center = (xmin + xmax) / 2.0 / img_width
        y_center = (ymin + ymax) / 2.0 / img_height
        w = (xmax - xmin) / img_width
        h = (ymax - ymin) / img_height
        lines.append(f"{cls_id} {x_center} {y_center} {w} {h}")

    os.makedirs(os.path.dirname(yolo_txt_path), exist_ok=True)
    with open(yolo_txt_path, "w") as f:
        f.write("\n".join(lines))

# Safe file copy if both image and annotation exist
def safe_copy(img_path, ann_path, img_out_dir, lbl_out_dir):
    if not os.path.exists(img_path) or not os.path.exists(ann_path):
        print(f"⚠ Skipping {img_path} (missing pair)")
        return False
    shutil.copy(img_path, img_out_dir)
    yolo_label_path = os.path.join(lbl_out_dir, os.path.splitext(os.path.basename(img_path))[0] + ".txt")
    convert_xml_to_yolo(ann_path, yolo_label_path)
    return True

# Process each dataset split
for split, split_file in splits.items():
    img_out_dir = os.path.join(mini_base, "images", split)
    lbl_out_dir = os.path.join(mini_base, "labels", split)
    os.makedirs(img_out_dir, exist_ok=True)
    os.makedirs(lbl_out_dir, exist_ok=True)

    with open(split_file, "r") as f:
        lines = f.read().splitlines()

    sample_size = math.ceil(len(lines) / 10)  # Take only 1/10th
    chosen = random.sample(lines, sample_size)

    count = 0
    for rel_path in chosen:
        # Expected paths
        img_path = os.path.join(base_images_dir, rel_path + ".jpg")
        ann_path = os.path.join(base_ann_dir, rel_path + ".xml")

        # If not found, search recursively
        if not os.path.exists(img_path):
            for root_dir, _, files in os.walk(base_images_dir):
                candidate = os.path.join(root_dir, os.path.basename(rel_path) + ".jpg")
                if os.path.exists(candidate):
                    img_path = candidate
                    break

        if not os.path.exists(ann_path):
            for root_dir, _, files in os.walk(base_ann_dir):
                candidate = os.path.join(root_dir, os.path.basename(rel_path) + ".xml")
                if os.path.exists(candidate):
                    ann_path = candidate
                    break

        if safe_copy(img_path, ann_path, img_out_dir, lbl_out_dir):
            count += 1

    print(f"✅ {split}: {count} valid samples processed.")

# Save YOLO data.yaml
yaml_content = f"""
train: {os.path.abspath(mini_base)}/images/train
val: {os.path.abspath(mini_base)}/images/val
test: {os.path.abspath(mini_base)}/images/test

nc: {len(classes)}
names: {classes}
"""
with open("data.yaml", "w") as f:
    f.write(yaml_content)

print("🎯 Mini YOLO dataset created and data.yaml saved successfully!")
