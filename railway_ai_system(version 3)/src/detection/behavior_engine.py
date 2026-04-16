import math

class BehaviorEngine:
    """
    Temporal Classifer utilizing geometric heuristics atop YOLO-Pose keypoints.
    Prioritizes: TAMPERING > PERSON_ON_TRACK > PERSON_NEAR_TRACK > SAFE_TRACK
    """
    def __init__(self, buffer_size=4):
        self.temporal_buffer = []
        self.buffer_size = buffer_size
        
        self.severity_map = {
            "TAMPERING_ACTIVITY_ALERT": 4,
            "PERSON_ON_TRACK_ALERT": 3,
            "PERSON_NEAR_TRACK_SAFE": 2,
            "SAFE_TRACK": 1
        }
    
    def evaluate_frame(self, human_detections, defect_detections, img_width=1920, img_height=1080):
        # 1. Base State Calculation
        raw_label = "SAFE_TRACK"
        
        if len(human_detections) == 0:
            raw_label = "SAFE_TRACK"
        else:
            max_severity = 1
            best_label = "SAFE_TRACK"
            
            for hd in human_detections:
                person_label = "PERSON_NEAR_TRACK_SAFE"
                
                # Extract Box
                hx1, hy1, hx2, hy2 = hd["bbox"]
                width, height = hx2 - hx1, hy2 - hy1
                cx, cy = (hx1 + hx2)/2, (hy1 + hy2)/2
                
                # Extract Keypoints
                keypoints = hd.get("keypoints", [])
                
                # --- HEURISTIC A: Posture (Crouching / Bending) ---
                is_crouching = False
                if width > height * 0.85: 
                    is_crouching = True  # wide bounding box => squat
                    
                if len(keypoints) >= 13:
                    try:
                        avg_shoulder_y = (keypoints[5][1] + keypoints[6][1]) / 2
                        avg_hip_y = (keypoints[11][1] + keypoints[12][1]) / 2
                        # Compression mapping: shoulders drop near hips means bending over
                        if abs(avg_hip_y - avg_shoulder_y) < (height * 0.3):
                            is_crouching = True
                    except: pass

                # --- HEURISTIC B: Spatial Proximity (On Track / Touching) ---
                touching_track = False
                interaction = False
                
                track_zone_x1, track_zone_x2 = img_width * 0.3, img_width * 0.7
                track_zone_y1, track_zone_y2 = img_height * 0.5, img_height
                
                if defect_detections:
                    for dd in defect_detections:
                        dcx, dcy, dw, dh_box = dd["obb_xywhr"][:4]
                        dx1, dy1 = dcx - dw/2, dcy - dh_box/2
                        dx2, dy2 = dcx + dw/2, dcy + dh_box/2
                        
                        # Box Overlap Calculation
                        if not (hx2 < dx1 or hx1 > dx2 or hy2 < dy1 or hy1 > dy2):
                            touching_track = True
                            
                            # Wrist to track object analysis
                            if len(keypoints) >= 11:
                                try:
                                    lw_x, lw_y = keypoints[9]
                                    rw_x, rw_y = keypoints[10]
                                    # Wrists inside the defect zone implies manual interaction
                                    if (dx1 - 40 < lw_x < dx2 + 40 and dy1 - 40 < lw_y < dy2 + 40) or \
                                       (dx1 - 40 < rw_x < dx2 + 40 and dy1 - 40 < rw_y < dy2 + 40):
                                        interaction = True
                                except: pass
                else:
                    # Generic Track Projection Area
                    if cy > track_zone_y1 and track_zone_x1 < cx < track_zone_x2:
                        person_label = "PERSON_NEAR_TRACK_SAFE"
                        if hy2 > track_zone_y2 * 0.8: # Deep inside track rails
                            touching_track = True
                            
                    # Generic Tool Verification
                    if is_crouching and len(keypoints) >= 17:
                        try:
                            avg_wrist_y = (keypoints[9][1] + keypoints[10][1]) / 2
                            avg_ankle_y = (keypoints[15][1] + keypoints[16][1]) / 2
                            # Wrists near ankles while bent signifies structural manipulation
                            if abs(avg_wrist_y - avg_ankle_y) < (height * 0.2):
                                interaction = True
                        except: pass
                
                # --- APPLY USER CLASSIFICATION RULES ---
                if touching_track:
                    person_label = "PERSON_ON_TRACK_ALERT"
                    
                if interaction or (is_crouching and touching_track):
                    person_label = "TAMPERING_ACTIVITY_ALERT"
                    
                # Strict Rules Resolution
                severity = self.severity_map.get(person_label, 1)
                if severity > max_severity:
                    max_severity = severity
                    best_label = person_label
                    
            raw_label = best_label
            
        # 2. Temporal Consistency Logic
        self.temporal_buffer.append(raw_label)
        if len(self.temporal_buffer) > self.buffer_size:
            self.temporal_buffer.pop(0)
            
        smoothed_label = "SAFE_TRACK"
        highest_sev = 1
        
        counts = {}
        for lbl in self.temporal_buffer:
            counts[lbl] = counts.get(lbl, 0) + 1
            
        for lbl, count in counts.items():
            # If label appears at least twice in buffer window, it's considered stable
            if count >= 2: 
                sev = self.severity_map[lbl]
                if sev > highest_sev:
                    highest_sev = sev
                    smoothed_label = lbl
                    
        # Fallback for empty buffer
        if highest_sev == 1 and len(self.temporal_buffer) < 2:
            smoothed_label = raw_label
            
        return smoothed_label
