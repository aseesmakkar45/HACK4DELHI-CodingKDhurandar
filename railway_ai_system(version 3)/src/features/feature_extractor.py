import numpy as np

class FeatureExtractor:
    """Converts raw YOLO detection attributes into normalized statistical features for ML."""
    
    def __init__(self, config_path="config.yaml"):
        # Eventually load the saved MinMaxScaler here if available
        self.total_features = 10
        
    def extract_frame_features(self, frame_data: dict, img_width=1920, img_height=1080):
        """
        Extracts 1D feature array (len 10) from single frame YOLO results.
        """
        detections = frame_data.get("detections", [])
        
        obj_count = len(detections)
        
        if obj_count == 0:
            return np.zeros(self.total_features).tolist()
            
        # Center proximity rule (how close is object to center track area)
        center_x, center_y = img_width / 2, img_height / 2
        proximities = []
        for d in detections:
            if d.get("type") == "track_defect":
                # OBB: [cx, cy, w, h, r]
                box_cx, box_cy = d["obb_xywhr"][0], d["obb_xywhr"][1]
                d["area"] = d["obb_xywhr"][2] * d["obb_xywhr"][3]
            elif d.get("type") == "human_tampering":
                bbox = d["bbox"]
                box_cx = (bbox[0] + bbox[2]) / 2
                box_cy = (bbox[1] + bbox[3]) / 2
                d["area"] = float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
            else:
                box_cx, box_cy, d["area"] = 0, 0, 0
                
            dist = np.sqrt((box_cx - center_x)**2 + (box_cy - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            proximities.append(1 - (dist / max_dist)) # 1.0 is dead center
            
        confs = [d["confidence"] for d in detections]
        areas = [d["area"] / (img_width * img_height) for d in detections] # Normalized area
        
        max_conf = max(confs)
        avg_conf = sum(confs) / obj_count
        conf_std = float(np.std(confs)) if obj_count > 1 else 0.0
        
        max_area = max(areas)
        avg_area = sum(areas) / obj_count
        
        # Risk thresholds
        anomalies = [d for d in detections if d["confidence"] > 0.6]
        anomaly_count = len(anomalies)
        anomaly_ratio = anomaly_count / obj_count if obj_count > 0 else 0.0
            
        max_center_prox = max(proximities)
        
        features = [
            float(obj_count),
            float(anomaly_count),
            float(anomaly_ratio),
            float(max_conf),
            float(avg_conf),
            float(conf_std),
            float(max_area),
            float(avg_area),
            float(max_center_prox),
            float(detections[0]["class_id"]) # dominant class placeholder
        ]
        
        return features
