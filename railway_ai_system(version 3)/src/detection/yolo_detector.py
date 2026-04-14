import os
import yaml
from ultralytics import YOLO

class YOLODetector:
    """Wrapper for Ultralytics YOLOv8 inference."""
    
    def __init__(self, config_path="config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
            
        self.device = self.config["yolo"].get("device", "cuda") # 4060 GPU
        self.conf_thresh = self.config["yolo"].get("confidence_threshold", 0.25)
        self.iou_thresh = self.config["yolo"].get("iou_threshold", 0.45)
        
        # 1. Defect Model (OBB)
        # Pointing to the local models folder inside practice project to guarantee portability
        mode_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models", "yolo_railway", "weights", "best.pt")
        if not os.path.exists(mode_path):
            print(f"WARNING: Model {mode_path} not found. Using default yolov8n-obb.pt")
            self.defect_model = YOLO('yolov8n-obb.pt')
        else:
            print(f"Loaded Custom YOLO-OBB Model: {mode_path}")
            self.defect_model = YOLO(mode_path)
            
        # 2. Tampering Model (Pose) — Required by BehaviorEngine for skeletal keypoint analysis
        print("Loading YOLO-Pose for Human Tampering Detection...")
        self.pose_model = YOLO('yolov8n-pose.pt')
            
        self.defect_model.to(self.device)
        self.pose_model.to(self.device)
        
    def analyze_batch(self, frame_paths: list, run_id: str):
        """
        Runs dual YOLO inference (OBB + Pose) on the batch.
        """
        print(f"[{run_id}] Running Dual-YOLOv8 Inference (OBB & Pose) on {len(frame_paths)} frames...")
        
        # 1. Defect Detection (OBB)
        defect_results = self.defect_model.predict(
            source=frame_paths, conf=self.conf_thresh, iou=self.iou_thresh,
            device=self.device, verbose=False, stream=False
        )
        
        # 2. Tampering Detection (Pose)
        pose_results = self.pose_model.predict(
            source=frame_paths, conf=self.conf_thresh, iou=self.iou_thresh,
            device=self.device, verbose=False, stream=False
        )
        
        processed_data = []
        
        for idx in range(len(frame_paths)):
            d_res = defect_results[idx]
            p_res = pose_results[idx]
            orig_h, orig_w = p_res.orig_shape
            
            frame_data = {
                "frame_path": frame_paths[idx],
                "img_width": orig_w,
                "img_height": orig_h,
                "defects_found": len(d_res.obb) if d_res.obb is not None else 0,
                "humans_found": len(p_res.boxes) if p_res.boxes is not None else 0,
                "detections": []
            }
            
            # Parse OBB (Oriented Bounding Boxes)
            if d_res.obb is not None:
                for obb in d_res.obb:
                    cls_id = int(obb.cls[0].item())
                    frame_data["detections"].append({
                        "type": "track_defect",
                        "class_id": cls_id,
                        "class_name": d_res.names[cls_id],
                        "confidence": float(obb.conf[0].item()),
                        # OBB representation: cx, cy, w, h, rotation
                        "obb_xywhr": obb.xywhr[0].tolist() 
                    })
                    
            # Parse Pose (Human keypoints for BehaviorEngine crouching/tampering detection)
            if p_res.boxes is not None:
                for idx_box, box in enumerate(p_res.boxes):
                    cls_id = int(box.cls[0].item())
                    
                    # Pose model classes: 0 is Person
                    if cls_id == 0:
                        keypoints = []
                        if p_res.keypoints is not None:
                            # 17 keypoints per person (shoulders, hips, wrists, ankles etc.)
                            keypoints = p_res.keypoints.xy[idx_box].tolist()
                            
                        frame_data["detections"].append({
                            "type": "human_tampering",
                            "class_id": cls_id,
                            "class_name": p_res.names[cls_id],
                            "confidence": float(box.conf[0].item()),
                            "bbox": box.xyxy[0].tolist(),
                            "keypoints": keypoints
                        })
                
            processed_data.append(frame_data)
            
        return processed_data
