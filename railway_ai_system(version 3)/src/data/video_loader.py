import os
import cv2
import yaml
from pathlib import Path

class VideoLoader:
    """Handles parsing video files into extracted frames for batch processing."""
    
    def __init__(self, config_path="config.yaml"):
        # Load configuration
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)
        
        self.frame_skip = self.config["video"].get("frame_skip", 3)
        self.output_base = Path("data/raw/frames/")
        
    def extract_frames(self, video_path: str, run_id: str):
        """
        Extracts frames from a video file into a Run-ID specific directory.
        Returns a list of extracted frame file paths and total frames evaluated.
        """
        run_output_dir = self.output_base / run_id
        os.makedirs(run_output_dir, exist_ok=True)
        
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found at {video_path}")
            
        if video_path.lower().endswith(('.jpg', '.jpeg', '.png', '.webp')):
            # Direct Image Analysis Logic
            print(f"[{run_id}] Loading static image file.")
            frame = cv2.imread(video_path)
            if frame is None:
                raise ValueError(f"Failed to load image: {video_path}")
            frame_filepath = run_output_dir / "frame_000000.jpg"
            cv2.imwrite(str(frame_filepath), frame)
            return [str(frame_filepath)], 1
            
        # Standard Video Analysis Logic
        cap = cv2.VideoCapture(video_path)
        frame_paths = []
        
        frame_idx = 0
        saved_idx = 0
        total_frames_in_video = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        print(f"[{run_id}] Loading video: {total_frames_in_video} total frames @ {fps} FPS")
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Only save frames based on frame_skip configurations
            if frame_idx % self.frame_skip == 0:
                frame_filename = f"frame_{saved_idx:06d}.jpg"
                frame_filepath = run_output_dir / frame_filename
                
                # Save physical file to disk
                cv2.imwrite(str(frame_filepath), frame)
                frame_paths.append(str(frame_filepath))
                saved_idx += 1
                
            frame_idx += 1

        cap.release()
        print(f"[{run_id}] Extraction complete. Saved {len(frame_paths)} frames.")
        
        return frame_paths, total_frames_in_video
