import os
import json
import time
import math
import joblib
from pathlib import Path
from src.data.video_loader import VideoLoader
from src.detection.yolo_detector import YOLODetector
from src.features.feature_extractor import FeatureExtractor
from src.detection.behavior_engine import BehaviorEngine

def run_analysis_pipeline(video_path: str, run_id: str):
    """
    End-to-End inference pipeline that coordinates:
    1. Video extraction to local frames
    2. YOLO Object Detection over all frames
    3. Structural statistical feature extraction
    4. Saving final run results
    """
    print(f"\n=====================================")
    print(f"| STARTING PIPELINE RUN: {run_id} |")
    print(f"=====================================\n")
    
    results_dir = Path(f"results/{run_id}")
    os.makedirs(results_dir, exist_ok=True)
    
    # Initialize Nodes
    loader = VideoLoader()
    detector = YOLODetector()
    extractor = FeatureExtractor()
    
    try:
        # Phase 1: Data Ingestion
        print("\n--> [Phase 1: Ingestion]")
        frame_paths, total_frames = loader.extract_frames(video_path, run_id)
        
        if not frame_paths:
            raise ValueError("Extraction yielded 0 frames. Aborting.")
            
        # Phase 2: Object Detection
        print("\n--> [Phase 2: YOLO Detection]")
        yolo_outputs = detector.analyze_batch(frame_paths, run_id)
        
        # Phase 3: Feature Assembly & ML Scoring
        print(f"\n--> [Phase 3: Classical ML Ensemble Scoring]")
        max_risk = 0.0
        global_anomalies = 0
        global_humans = 0
        
        # Load Classical ML Models dynamically
        ml_models = {}
        try:
            ml_models["knn"] = joblib.load("models/ml/knn.pkl")
            ml_models["logreg"] = joblib.load("models/ml/logreg.pkl")
            ml_models["dtree"] = joblib.load("models/ml/dtree.pkl")
            ml_models["rforest"] = joblib.load("models/ml/rforest.pkl")
        except Exception as e:
            print("WARNING: Could not load Scikit-Learn models. Ensure you ran train_ml_layer.py!")
            
        final_timeline_data = []
        scikit_scores = { "knn": 0.0, "logreg": 0.0, "dtree": 0.0, "rforest": 0.0 }
        peak_scikit_scores = { "knn": 0.0, "logreg": 0.0, "dtree": 0.0, "rforest": 0.0 }
        
        # Threat Tracking Memory Engine
        behavior_engine = BehaviorEngine(buffer_size=5)
        
        for frame_data in yolo_outputs:
            # Generate ML 1D feature array
            feature_vector = extractor.extract_frame_features(frame_data, img_width=1920, img_height=1080)
            
            # Scikit-Learn Layer 2 Prediction
            if len(ml_models) == 4:
                # Reshape 1D array to 2D for sklearn
                X_input = [feature_vector]
                scikit_scores["knn"] = ml_models["knn"].predict_proba(X_input)[0][1]
                scikit_scores["logreg"] = ml_models["logreg"].predict_proba(X_input)[0][1]
                scikit_scores["dtree"] = ml_models["dtree"].predict_proba(X_input)[0][1]
                scikit_scores["rforest"] = ml_models["rforest"].predict_proba(X_input)[0][1]
                
                # The Base Risk is literally the statistical average of the Classical Ensembles
                fused_risk = sum(scikit_scores.values()) / 4.0
            else:
                # Fallback if models not found
                anomaly_ratio = feature_vector[2]
                max_conf = feature_vector[3]
                fused_risk = (anomaly_ratio * 0.6) + (max_conf * 0.4)
            
            # ----------------------------------------------------
            # Advanced Algorithmic Fusion Layer: Temporal Behavior
            # ----------------------------------------------------
            human_detections = [d for d in frame_data["detections"] if d.get("type") == "human_tampering"]
            defect_detections = [d for d in frame_data["detections"] if d.get("type") == "track_defect"]
            
            # Execute Temporal Heuristics 
            confirmed_behavior = behavior_engine.evaluate_frame(human_detections, defect_detections)
            
            # Dynamically scale the Fused Risk Score based on structural interactions
            if confirmed_behavior == "TAMPERING_ACTIVITY_ALERT":
                fused_risk += 0.8
            elif confirmed_behavior == "PERSON_ON_TRACK_ALERT":
                fused_risk += 0.4
            elif confirmed_behavior == "PERSON_NEAR_TRACK_SAFE":
                fused_risk += 0.1
                
            # Ensure risk is capped between 0 and 1
            fused_risk = min(1.0, fused_risk)
            
            if fused_risk > max_risk:
                max_risk = fused_risk
                peak_scikit_scores = dict(scikit_scores)
                
            global_anomalies += feature_vector[1] # raw anomaly count
            global_humans += frame_data.get("humans_found", 0)
            
            final_timeline_data.append({
                "frame": frame_data["frame_path"],
                "img_width": frame_data.get("img_width", 1920),
                "img_height": frame_data.get("img_height", 1080),
                "yolo_raw": frame_data["detections"],
                "features": feature_vector,
                "fused_risk": fused_risk,
                "behavior_label": confirmed_behavior
            })
            
        # Phase 4: Summarize and Save
        summary_payload = {
            "run_id": run_id,
            "status": "complete",
            "video_processed_path": video_path,
            "total_frames_analyzed": len(frame_paths),
            "anomalies_detected": int(global_anomalies),
            "humans_detected": int(global_humans),
            "max_risk_score": float(max_risk),
            "scikit_ensemble_scores": peak_scikit_scores if len(ml_models) > 0 else {},
            "timeline": final_timeline_data
        }
        
        out_file = results_dir / "pipeline_output.json"
        with open(out_file, "w") as f:
            json.dump(summary_payload, f, indent=4)
            
        print(f"\n✅ Pipeline Complete! Results saved to: {out_file}")
        
    except Exception as e:
        print(f"\n❌ Pipeline Failed: {str(e)}")
        error_payload = {"run_id": run_id, "status": "failed", "error": str(e)}
        with open(results_dir / "pipeline_output.json", "w") as f:
            json.dump(error_payload, f)
