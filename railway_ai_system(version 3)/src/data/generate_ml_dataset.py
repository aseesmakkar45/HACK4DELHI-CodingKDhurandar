import os
import cv2
import csv
import math
import glob
import numpy as np
from src.detection.yolo_detector import YOLODetector
from src.features.feature_extractor import FeatureExtractor

def calculate_heuristic_risk(frame_data):
    """
    Simulates the exact physical logic from inference_pipeline.py to label our dataset
    with a perfectly deterministic Ground Truth "Risk Score" out of 1.0
    """
    human_detections = [d for d in frame_data["detections"] if d.get("type") == "human_tampering"]
    defect_detections = [d for d in frame_data["detections"] if d.get("type") == "track_defect"]
    
    anomaly_ratio = len(defect_detections) / max(1, len(frame_data["detections"]))
    max_conf = max([d.get("confidence", 0.0) for d in defect_detections], default=0.0)
    
    fused_risk = (anomaly_ratio * 0.6) + (max_conf * 0.4)
    
    critical_overlap = False
    for hd in human_detections:
        hx1, hy1, hx2, hy2 = hd["bbox"]
        for dd in defect_detections:
            cx, cy, w, h_box, r = dd["obb_xywhr"]
            dx1, dy1 = cx - w/2, cy - h_box/2
            dx2, dy2 = cx + w/2, cy + h_box/2
            if not (hx2 < dx1 or hx1 > dx2 or hy2 < dy1 or hy1 > dy2):
                critical_overlap = True
                break
                
    if critical_overlap:
        fused_risk += 0.5
    elif len(human_detections) > 0:
        fused_risk += 0.2
        
    return min(1.0, fused_risk)

def generate_dataset():
    print("="*60)
    print("🚂 RailGuard AI - Classical ML Data Generator")
    print("="*60)
    
    # Initialize Core Engines
    detector = YOLODetector()
    extractor = FeatureExtractor()
    
    # Locate dataset
    image_dir = os.path.join("train", "images")
    image_paths = glob.glob(os.path.join(image_dir, "*.jpg"))
    
    # Quick execution: We'll sample 500 images max to keep it blazing fast for Local Execution
    image_paths = image_paths[:500] 
    print(f"\n[1/3] Located {len(image_paths)} Ground Truth images for dataset extraction...")
    
    # Prepare CSV Output
    os.makedirs("data/ml_features", exist_ok=True)
    csv_file = "data/ml_features/railguard_features.csv"
    
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header (10 Features + 1 Target Label)
        writer.writerow(['obj_count', 'anomaly_count', 'anomaly_ratio', 'max_conf', 
                         'avg_conf', 'conf_std', 'max_area', 'avg_area', 
                         'max_center_prox', 'class_id', 'Risk_Score_Y'])
                         
        print(f"[2/3] Extracting Authentic YOLO Statistical Distributions...")
        for i, img_path in enumerate(image_paths):
            frame = cv2.imread(img_path)
            if frame is None: continue
            h, w = frame.shape[:2]
            
            # 1. Run YOLO CV
            frame_data = detector.analyze_batch([img_path], "dataset_gen")[0]
            
            # 2. Inject Contextual Entropy
            # To train the Scikit models to recognize humans tampering, we mathematically
            # inject a fake human bounding box onto 30% of the images since our dataset 
            # exclusively consists of deserted tracks.
            if np.random.rand() < 0.3:
                # Random human bounding box
                hx1, hy1 = int(w*np.random.rand()*0.5), int(h*np.random.rand()*0.5)
                frame_data["detections"].append({
                    "type": "human_tampering",
                    "bbox": [hx1, hy1, hx1+100, hy1+250],
                    "keypoints": [], # Ignored by AABB math
                    "confidence": float(round(0.6 + np.random.rand()*0.3, 2)),
                    "class_id": 0.0
                })
                frame_data["humans_found"] = frame_data.get("humans_found", 0) + 1
                
            # 3. Calculate Ground Truth Math Target Label (Y)
            risk_y = calculate_heuristic_risk(frame_data)
            
            # 4. Extract Real Feature Array (X)
            features_x = extractor.extract_frame_features(frame_data, w, h)
            
            # Write to CSV
            row = features_x + [risk_y]
            writer.writerow(row)
            
            if i % 100 == 0 and i > 0:
                print(f"      Processed {i}/{len(image_paths)} images...")
                
    print(f"\n[3/3] Feature extraction complete! Saved to: {csv_file}")
    
if __name__ == "__main__":
    generate_dataset()
