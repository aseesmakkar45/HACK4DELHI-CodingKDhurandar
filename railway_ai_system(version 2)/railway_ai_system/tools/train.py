"""
RailDrishti AI — Custom Model Training Pipeline
================================================
Fine-tunes YOLOv8-Pose on railway track tampering data.

Usage:
  python tools/train.py --data dataset/data.yaml --epochs 50 --model yolov8n-pose.pt

Output:
  runs/train/weights/best.pt  → drop into project root to replace the default model
"""

import argparse
import os
from pathlib import Path

try:
    from ultralytics import YOLO
except ImportError:
    print("Install ultralytics: pip install ultralytics")
    exit(1)


DEFAULT_MODEL  = "yolov8n-pose.pt"
DEFAULT_EPOCHS = 50
DEFAULT_IMGSZ  = 640
DEFAULT_BATCH  = 8
DEFAULT_DATA   = "dataset/data.yaml"


def train(model_path: str, data: str, epochs: int, imgsz: int, batch: int):
    print(f"\n{'='*60}")
    print(f"  RailDrishti AI — Custom Training Pipeline")
    print(f"{'='*60}")
    print(f"  Model  : {model_path}")
    print(f"  Dataset: {data}")
    print(f"  Epochs : {epochs}")
    print(f"  ImgSz  : {imgsz}")
    print(f"  Batch  : {batch}")
    print(f"{'='*60}\n")

    if not os.path.exists(data):
        print(f"[ERROR] Dataset config not found: {data}")
        print("  Run tools/download_dataset.py first, or manually place images in dataset/")
        return

    model = YOLO(model_path)

    results = model.train(
        data=data,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        project="runs",
        name="raildrishti_tamper",
        save=True,
        plots=True,
        verbose=True,
        # Transfer learning — freeze backbone, only train detection head
        freeze=10,
        # Data augmentation for railway-specific scenarios
        hsv_h=0.015,    # slight hue variation (lighting changes)
        hsv_s=0.4,      # saturation variation
        hsv_v=0.4,      # brightness variation (night/day)
        flipud=0.0,      # no vertical flip (cameras are fixed orientation)
        fliplr=0.5,      # horizontal flip is fine
        mosaic=1.0,      # mosaic augmentation for small datasets
        mixup=0.1,
        degrees=5.0,     # slight rotation (camera angle variation)
    )

    best_weights = Path("runs/raildrishti_tamper/weights/best.pt")
    if best_weights.exists():
        print(f"\n{'='*60}")
        print(f"  Training complete!")
        print(f"  Best weights saved to: {best_weights}")
        print(f"  To deploy: copy best.pt to project root.")
        print(f"  The server will load it automatically on next restart.")
        print(f"{'='*60}\n")
    else:
        print("[WARN] Training may have not completed. Check runs/ directory.")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RailDrishti AI — Custom Model Trainer")
    parser.add_argument("--model",  default=DEFAULT_MODEL,  help="Base model weights")
    parser.add_argument("--data",   default=DEFAULT_DATA,   help="dataset/data.yaml path")
    parser.add_argument("--epochs", default=DEFAULT_EPOCHS, type=int)
    parser.add_argument("--imgsz",  default=DEFAULT_IMGSZ,  type=int)
    parser.add_argument("--batch",  default=DEFAULT_BATCH,  type=int)
    args = parser.parse_args()

    train(args.model, args.data, args.epochs, args.imgsz, args.batch)
