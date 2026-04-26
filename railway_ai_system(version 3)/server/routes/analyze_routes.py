from fastapi import APIRouter, BackgroundTasks, UploadFile, File, Depends
from sqlalchemy.orm import Session
import uuid
import os
import shutil
import json
from src.pipeline.inference_pipeline import run_analysis_pipeline
from server.database import get_db, SessionLocal
from server.models import Run, Alert

router = APIRouter()

def execute_and_store(video_path: str, run_id: str):
    # Process Machine Learning physics
    run_analysis_pipeline(video_path, run_id)
    
    # Hook metrics into the Persistent Database
    db = SessionLocal()
    try:
        results_file = f"results/{run_id}/pipeline_output.json"
        if os.path.exists(results_file):
            with open(results_file, "r") as f:
                data = json.load(f)
                
            db_run = db.query(Run).filter(Run.id == run_id).first()
            if db_run:
                db_run.status = "complete"
                db_run.risk_score = data.get("max_risk_score", 0.0)
                db_run.anomalies_detected = data.get("anomalies_detected", 0)
                db_run.humans_detected = data.get("humans_detected", 0)
                db_run.total_frames = data.get("total_frames_analyzed", 0)
                
                # Launch automated security alerts
                if db_run.risk_score > 0.75 or db_run.humans_detected > 0:
                    alert = Alert(
                        id=f"alt_{uuid.uuid4().hex[:8]}",
                        run_id=run_id,
                        message=f"CRITICAL TAMPERING: {db_run.humans_detected} unauthorized entities and {db_run.anomalies_detected} anomalies detected. Fusion Risk: {db_run.risk_score * 100:.1f}%",
                        severity="high"
                    )
                    db.add(alert)
                    
            db.commit()
    except Exception as e:
        print("Database Serialization Error:", e)
    finally:
        db.close()

@router.post("")
def trigger_analysis(background_tasks: BackgroundTasks, file: UploadFile = File(None), db: Session = Depends(get_db)):
    # Generate unique ID for this analysis run
    run_id = f"run_{uuid.uuid4().hex[:8]}"
    
    video_name = "test_image"
    # Physical file upload logic
    if file:
        video_name = file.filename
        os.makedirs("data/raw/videos", exist_ok=True)
        video_path = f"data/raw/videos/{run_id}_{file.filename}"
        with open(video_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    else:
        # Fallback mock if no file provided
        video_path = r"C:\Users\lenovo\Desktop\practice project\train\images\Image-001_jpg.rf.21d659ac7cc76b3de75db53e84346957.jpg"
        
    # Queue into Database Ledger before processing starts
    new_run = Run(id=run_id, video_name=video_name, status="processing")
    db.add(new_run)
    db.commit()
        
    # Queue the heavy python pipeline to run asynchronously through the DB Hook
    background_tasks.add_task(execute_and_store, video_path, run_id)
    
    return {"status": "started", "run_id": run_id}

@router.get("/{run_id}/progress")
def get_progress(run_id: str):
    import os
    import json
    
    results_file = f"results/{run_id}/pipeline_output.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            data = json.load(f)
            if data.get("status") == "failed":
                return {"run_id": run_id, "status": "failed", "progress": 0}
        return {"run_id": run_id, "status": "complete", "progress": 100}
        
    # Since we don't have atomic progress tracking yet, return a placeholder
    # while the batch is still running in the background.
    return {"run_id": run_id, "status": "processing", "progress": 50}
