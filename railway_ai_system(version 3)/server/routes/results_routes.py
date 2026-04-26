from fastapi import APIRouter
import os
import json

router = APIRouter()

@router.get("/ml/feature-importance")
def get_feature_importance():
    importance_file = "models/ml/feature_importance.json"
    if os.path.exists(importance_file):
        with open(importance_file, "r") as f:
            return json.load(f)
    return [{"name": "waiting on training", "value": 0}]

@router.get("/{run_id}")
def get_results(run_id: str):
    results_file = f"results/{run_id}/pipeline_output.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            data = json.load(f)
            
            # Since the frontend UI AnalysisResults.jsx expects specific keys,
            timeline = data.get("timeline", [])
            
            # --- INTELLIGENCE EXTRACTION ---
            class_counts = {}
            behavior_priority = {
                "TAMPERING_ACTIVITY_ALERT": 3,
                "PERSON_ON_TRACK_ALERT": 2,
                "PERSON_NEAR_TRACK_SAFE": 1,
                "TRACK_SAFE": 0
            }
            highest_behavior_score = -1
            primary_behavior = "TRACK_SAFE"
            
            for frame in timeline:
                for detection in frame.get("yolo_raw", []):
                    cls_name = detection.get("class_name", "unknown")
                    class_counts[cls_name] = class_counts.get(cls_name, 0) + 1
                    
                label = frame.get("behavior_label", "TRACK_SAFE")
                b_score = behavior_priority.get(label, 0)
                if b_score > highest_behavior_score:
                    highest_behavior_score = b_score
                    primary_behavior = label
            
            true_dominant_class = "benign_activity"
            if class_counts:
                # Filter out 'normal'/'train' etc if there are actual threats (basic heuristic)
                threats = {k: v for k, v in class_counts.items() if k not in ["normal", "safe_track", "train"]}
                if threats:
                    true_dominant_class = max(threats, key=threats.get)
                else:
                    true_dominant_class = max(class_counts, key=class_counts.get)
            
            # --- TEMPORAL INCIDENT CLUSTERING ENGINE ---
            clustered_events = []
            current_event = None
            
            for i, frame in enumerate(timeline):
                f_risk = frame.get("fused_risk", 0.0)
                b_label = frame.get("behavior_label", "TRACK_SAFE")
                
                # Check Danger Thresholds (>0.5 risk or an active alert label)
                is_dangerous = f_risk > 0.5 or b_label in ["TAMPERING_ACTIVITY_ALERT", "PERSON_ON_TRACK_ALERT"]
                
                if is_dangerous:
                    if not current_event:
                        # Open new Incident
                        current_event = {
                            "incident_id": f"INC_{i}_{len(clustered_events)+1}",
                            "start_frame_idx": i,
                            "end_frame_idx": i,
                            "duration_frames": 1,
                            "event_type": b_label,
                            "peak_risk": f_risk,
                            "representative_frame": frame
                        }
                    else:
                        # Extend Incident
                        current_event["end_frame_idx"] = i
                        current_event["duration_frames"] = (i - current_event["start_frame_idx"]) + 1
                        
                        # Find the absolute peak moment to represent the cluster
                        if f_risk > current_event["peak_risk"]:
                            current_event["peak_risk"] = f_risk
                            current_event["event_type"] = b_label
                            current_event["representative_frame"] = frame 
                else:
                    if current_event:
                        clustered_events.append(current_event)
                        current_event = None
            
            if current_event:
                clustered_events.append(current_event)
            
            # --- FORMAT RESPONSE ---
            scores = data.get("scikit_ensemble_scores", {})
            ml_avg = sum(scores.values()) / max(len(scores), 1) if scores else 0.0
            behavior_boost = max(0.0, data.get("max_risk_score", 0.0) - ml_avg)
            
            return {
                "run_id": run_id,
                "risk_score": data.get("max_risk_score", 0.0),
                "dominant_class": true_dominant_class,
                "primary_behavior": primary_behavior,
                "models": [
                    { "name": "KNN", "score": scores.get("knn", 0.0) },
                    { "name": "LogReg", "score": scores.get("logreg", 0.0) },
                    { "name": "DTree", "score": scores.get("dtree", 0.0) },
                    { "name": "RForest", "score": scores.get("rforest", 0.0) },
                    { "name": "BehaviorEngine", "score": min(1.0, behavior_boost) }
                ],
                "total_frames_analyzed": data.get("total_frames_analyzed", 0),
                "anomalies_detected": data.get("anomalies_detected", 0),
                "humans_detected": data.get("humans_detected", 0),
                "incident_events": clustered_events
            }
            
    return {"run_id": run_id, "error": "Results not found or still processing."}

@router.get("/{run_id}/summary")
def get_summary(run_id: str):
    return {"run_id": run_id, "summary": "Sample summary"}

@router.get("/helper/retrain-models")
def retrain_ml_models():
    import sys
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.models.train_ml_layer import train_ensemble_layer
    try:
        train_ensemble_layer()
        return {"status": "success", "msg": "retrained using " + sys.executable}
    except Exception as e:
        return {"status": "error", "msg": str(e)}
