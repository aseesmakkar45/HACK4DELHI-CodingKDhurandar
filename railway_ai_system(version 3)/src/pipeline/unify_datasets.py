import os
import shutil
from pathlib import Path

# ==============================================================================
# RailGuard AI - Universal Dataset Merger
# ==============================================================================
# This scripts takes disparate datasets (Roboflow 5-class, Intrusion, Kaggle) 
# and safely merges them into a single YOLO-ready folder with unified classes!
# ==============================================================================

# THE MASTER TAXONOMY (Our Ultimate 10 Classes)
MASTER_CLASSES = {
    0: 'crack',
    1: 'break',
    2: 'scar',
    3: 'spalling',
    4: 'normal',
    5: 'person',        # From Intrusion dataset
    6: 'train',         # Context
    7: 'maintenance',   # Context
    8: 'defect_other',  # Catch-all
    9: 'safe_track'     # from Kaggle Background
}

# ==============================================================================
# STEP 1: DEFINE YOUR FOLDERS HERE
# Change "1", "2", "3" to match whichever folder holds which dataset
# ==============================================================================

DATASET_MAPPINGS = [
    {
        "name": "Roboflow_Defects",
        "path": "datasets/2",         # Identified: 5 classes, ~1100 images
        "class_map": {
            0: 0, 1: 1, 2: 2, 3: 3, 4: 4
        }
    },
    {
        "name": "Intrusion_Detection",
        "path": "datasets/1",         # Identified: 1 class (Person), ~2000 images
        "class_map": {
            0: 5 
        }
    },
    {
        "name": "Other_Defects_Fasteners",
        "path": "datasets/3",         # Identified: 1 class, ~5700 images (likely fasteners/other)
        "class_map": {
            0: 8 # Mapped to 'defect_other'
        }
    },
    {
        "name": "Background_Healthy_Tracks",
        "path": "datasets/4",         # Identified: 0 classes, ~360 images (Hard Negatives)
        "class_map": {}
    }
]

OUTPUT_DIR = Path("datasets/master_railguard")

def create_structure():
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    
    for split in ['train', 'valid']:
        os.makedirs(OUTPUT_DIR / split / 'images', exist_ok=True)
        os.makedirs(OUTPUT_DIR / split / 'labels', exist_ok=True)

def merge_dataset(config):
    base_path = Path(config["path"])
    print(f"\n🚀 Merging Dataset: {config['name']} from {base_path}")
    
    if not base_path.exists():
        print(f"  ❌ Skipping. Path {base_path} does not exist.")
        return

    for split in ['train', 'valid']:
        split_img_dir = base_path / split / 'images'
        split_lbl_dir = base_path / split / 'labels'
        
        if not split_img_dir.exists() or not split_lbl_dir.exists():
            # Sometimes folders are uppercase
            split_img_dir = base_path / split.capitalize() / 'images'
            split_lbl_dir = base_path / split.capitalize() / 'labels'
            if not split_img_dir.exists():
                print(f"  ⚠️ Could not find {split} split in {config['name']}")
                continue

        images = list(split_img_dir.glob("*.jpg")) + list(split_img_dir.glob("*.jpeg")) + list(split_img_dir.glob("*.png"))
        print(f"  --> Processing {len(images)} files in {split}...")

        for img_path in images:
            # Check for label
            lbl_path = split_lbl_dir / (img_path.stem + ".txt")
            
            # Destination path (prefix with dataset name to avoid name collisions!)
            dest_prefix = f"{config['name']}_{img_path.stem}"
            dest_img = OUTPUT_DIR / split / 'images' / f"{dest_prefix}{img_path.suffix}"
            dest_lbl = OUTPUT_DIR / split / 'labels' / f"{dest_prefix}.txt"

            shutil.copy2(img_path, dest_img)

            if lbl_path.exists():
                with open(lbl_path, 'r') as f_in, open(dest_lbl, 'w') as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if not parts: continue
                        
                        orig_cls = int(parts[0])
                        # Map to Master Class if mapping exists, otherwise ignore/keep as 8 (other)
                        if orig_cls in config["class_map"]:
                            new_cls = config["class_map"][orig_cls]
                            parts[0] = str(new_cls)
                            f_out.write(" ".join(parts) + "\n")
            else:
                # Background image (No bounding boxes). Just create an empty txt file
                dest_lbl.touch()

def write_master_yaml():
    yaml_content = f"""path: {os.path.abspath(OUTPUT_DIR)}
train: train/images
val: valid/images
test: test/images

nc: {len(MASTER_CLASSES)}
names: {[MASTER_CLASSES[i] for i in range(len(MASTER_CLASSES))]}
"""
    with open(OUTPUT_DIR / "data.yaml", "w") as f:
        f.write(yaml_content)
    print("\n✅ Created master data.yaml!")

if __name__ == "__main__":
    print("="*60)
    print("🚂 RailGuard AI - Dataset Merger Engine Initialize")
    print("="*60)
    
    create_structure()
    for ds in DATASET_MAPPINGS:
        merge_dataset(ds)
        
    write_master_yaml()
    print("\n🎉 DONE! All datasets unified under datasets/master_railguard/")
    print("Update train_yolo.py to point to datasets/master_railguard/data.yaml")
