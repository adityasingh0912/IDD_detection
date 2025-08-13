import os
import sys
import subprocess

def run_inference(
    weights_path="yolov5/runs/train/exp2/weights/best.pt",
    source_dir="mini_dataset/images/test",
    conf_thresh=0.25,
    output_project="final_results",
    output_name="predictions"
):
    yolov5_detect = os.path.join("yolov5", "detect.py")
    cmd = [
        sys.executable, yolov5_detect,   # ✅ Use current Python interpreter
        "--weights", weights_path,
        "--source", source_dir,
        "--conf", str(conf_thresh),
        "--project", output_project,
        "--name", output_name,
        "--exist-ok"
    ]
    subprocess.run(cmd)

if __name__ == "__main__":
    run_inference()
    print("\n✅ Inference complete! Check results in 'final_results/predictions/'.")
