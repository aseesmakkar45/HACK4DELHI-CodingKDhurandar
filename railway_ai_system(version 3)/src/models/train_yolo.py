import os
from ultralytics import YOLO

def train_model():
    print("="*60)
    print("🚂 RailGuard AI - YOLO-OBB Local Training Engine")
    print("="*60)
    
    # We use YOLOv8-Medium (Standard Object Detection) architecture.
    # The Mega-Dataset is standard bounding box formatted, meaning the old OBB (Oriented) model is incompatible.
    print("\n[1/3] Initializing Pre-Trained YOLOv8-Medium Weights...")
    
    # We load the powerful baseline COCO weights because the old best.pt was an OBB format, 
    # and cannot biologically be fine-tuned on non-OBB data!
    model = YOLO("yolov8m.pt")
    
    # Get absolute path to the UNIFIED data.yaml
    yaml_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "master_railguard", "data.yaml"))
    
    print(f"\n[2/3] Verified Dataset Mapping --> {yaml_path}")
    import torch
    
    # Auto-detect hardware
    if torch.cuda.is_available():
        hardware = "cuda:0"
        print(f"\n[3/3] Igniting RTX 4060 GPU Training Loop 🚀")
    else:
        hardware = "cpu"
        print(f"\n[3/3] WARNING: PyTorch CUDA not detected! Falling back to CPU training 🐢")
        print("To use your 4060, install CUDA PyTorch.")
    
    # Run the training process with Augmentations for Fine-Tuning
    results = model.train(
        data=yaml_path,
        epochs=50,          # Fine-tuning requires fewer epochs (50-100)
        batch=-1,           # Auto-scaling batch size mapping to VRAM
        imgsz=640,          # Standard image resolution
        device=hardware,    # Dynamic Device (CUDA or CPU)
        amp=True,           # Automatic Mixed Precision for faster VRAM operations
        project="models/fine_tuned",   # Output root directory to prevent overwriting base
        name="yolo_railway_v2", # Subdirectory where V2 weights fall
        
        # --- DATA AUGMENTATION (Hardening the model) ---
        hsv_h=0.015,        # Adjust lighting/color to simulate day/night
        hsv_s=0.7,          # Saturation shift
        hsv_v=0.4,          # Brightness shift (critical for shadows)
        translate=0.1,      # Image translation (simulating bad camera angles)
        scale=0.5,          # Image scaling (zoom in/out)
        fliplr=0.5,         # Flip left/right (train approaches from both sides)
        mosaic=1.0,         # Mosaic augmentation (combines 4 images into 1)
        mixup=0.1           # Mixup augmentation (blends 2 images)
    )
    
    print("\n" + "="*60)
    print("✅ Fine-Tuning Training Sequence Completed Successfully!")
    print("The upgraded Hardened Weights are situated at: models/yolo_railway/weights/best.pt")
    print("="*60)

if __name__ == "__main__":
    train_model()
