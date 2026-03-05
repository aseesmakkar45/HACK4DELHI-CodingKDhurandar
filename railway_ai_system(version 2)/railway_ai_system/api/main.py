import cv2
import asyncio
import json
import os
import time
import random
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from ultralytics import YOLO
import threading

import api.db as db
import api.response as response_dispatcher
import api.demo as demo_mode

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
#  MODEL — YOLOv8-Pose gives us skeleton keypoints per person
# ─────────────────────────────────────────────────────────────
try:
    # yolov8n-pose.pt is auto-downloaded on first run (~7 MB)
    model = YOLO("yolov8n-pose.pt")
    MODEL_MODE = "pose"
except Exception:
    try:
        model = YOLO("yolov8n.pt")
        MODEL_MODE = "detect"
    except Exception:
        model = None
        MODEL_MODE = "none"

# ─────────────────────────────────────────────────────────────
#  GLOBAL STATE
# ─────────────────────────────────────────────────────────────
clients = set()
latest_frame = None
latest_frame_condition = threading.Condition()
alarm_queue = asyncio.Queue()
main_loop = None
maintenance_mode = {"active": False, "until": 0}   # global maintenance flag

# ─────────────────────────────────────────────────────────────
#  TEMPORAL BEHAVIOUR ENGINE (TBE)
#  Tracks each detected person across frames and computes a
#  tamper_confidence_score based on posture + dwell time.
# ─────────────────────────────────────────────────────────────
class PersonTracker:
    """
    Tracks a single detected person's behaviour across consecutive frames.

    Scoring logic:
      +0.15  per frame the person is CROUCHING  (hip keypoint near ankle)
      +0.10  per frame the person is STATIONARY (centroid ≤ 15px drift)
      +0.05  per frame hands are AT TRACK LEVEL (wrist y > hip y)
      −0.20  per frame the person is WALKING fast (centroid > 40px drift)
      −0.30  applied instantly when the person leaves the frame

    Alarm fires when tamper_confidence_score ≥ 0.75
    """
    CROUCH_THRESHOLD   = 0.70   # ratio: hip_y / ankle_y — below this = crouching
    STATIONARY_DRIFT   = 18     # pixels — movement budget considered "stationary"
    WALK_DRIFT         = 45     # pixels — above this = clearly passing through
    ALARM_THRESHOLD    = 0.75   # tamper_confidence to trigger alert
    MAX_SCORE          = 1.0
    MIN_SCORE          = 0.0
    DECAY_PER_FRAME    = 0.03   # score bleeds down when nothing suspicious

    def __init__(self, person_id):
        self.person_id   = person_id
        self.score       = 0.0
        self.prev_cx     = None
        self.prev_cy     = None
        self.frames_seen = 0
        self.alarmed     = False

    def update(self, cx, cy, keypoints):
        """
        cx, cy      – centroid of bounding box
        keypoints   – list of (x, y, confidence) from YOLOv8-Pose
                      COCO order: 0=nose … 11=left_hip, 12=right_hip,
                                  13=left_knee, 14=right_knee,
                                  15=left_ankle, 16=right_ankle,
                                   9=left_wrist, 10=right_wrist
        Returns tamper_confidence_score (float 0–1)
        """
        self.frames_seen += 1
        delta = 0.0

        # ── Drift (movement) analysis ──────────────────────────
        drift = 0
        if self.prev_cx is not None:
            drift = ((cx - self.prev_cx)**2 + (cy - self.prev_cy)**2) ** 0.5

        self.prev_cx, self.prev_cy = cx, cy

        if drift > self.WALK_DRIFT:
            # Person is walking past — penalise
            delta -= 0.20
        elif drift < self.STATIONARY_DRIFT:
            # Person is stationary — suspicious
            delta += 0.10

        # ── Posture analysis from keypoints ───────────────────
        if keypoints is not None and len(keypoints) >= 17:
            def kp(idx):
                """Return (x, y) if confidence > 0.4 else None."""
                k = keypoints[idx]
                if len(k) >= 3 and float(k[2]) > 0.4:
                    return float(k[0]), float(k[1])
                return None

            # Average hip and ankle positions
            left_hip    = kp(11); right_hip  = kp(12)
            left_ankle  = kp(15); right_ankle= kp(16)
            left_wrist  = kp(9);  right_wrist= kp(10)
            left_shoulder=kp(5);  right_shoulder=kp(6)

            hip_y   = np.mean([p[1] for p in [left_hip, right_hip]     if p]) if any([left_hip, right_hip])     else None
            ankle_y = np.mean([p[1] for p in [left_ankle, right_ankle] if p]) if any([left_ankle, right_ankle]) else None
            wrist_y = np.mean([p[1] for p in [left_wrist, right_wrist] if p]) if any([left_wrist, right_wrist]) else None
            shoulder_y = np.mean([p[1] for p in [left_shoulder, right_shoulder] if p]) if any([left_shoulder, right_shoulder]) else None

            # Crouching: hip is close to ankle (ratio > threshold)
            if hip_y and ankle_y and ankle_y > 0:
                crouch_ratio = hip_y / ankle_y
                if crouch_ratio > self.CROUCH_THRESHOLD:
                    delta += 0.15   # definitely crouching

            # Hands at track level: wrists are lower than hips
            if wrist_y is not None and hip_y is not None and wrist_y > hip_y:
                delta += 0.05       # hands down near track

            # Bent-over posture: shoulder close to hip
            if shoulder_y is not None and hip_y is not None:
                torso_ratio = abs(shoulder_y - hip_y)
                if torso_ratio < 40:   # pixels — very bent
                    delta += 0.08

        # ── Natural decay — score doesn't stay high without cause ─
        delta -= self.DECAY_PER_FRAME

        self.score = max(self.MIN_SCORE, min(self.MAX_SCORE, self.score + delta))
        return self.score

    def is_tamper_alarm(self):
        return (not self.alarmed) and (self.score >= self.ALARM_THRESHOLD)

    def mark_alarmed(self):
        self.alarmed = True


# One tracker per person bounding box (keyed by a simple proximity hash)
_person_trackers = {}   # key: str(approx_cx_cy) → PersonTracker

def _find_or_create_tracker(cx, cy):
    """Find existing tracker within 80px, else create new one."""
    for key, tracker in _person_trackers.items():
        if tracker.prev_cx is not None:
            dist = ((tracker.prev_cx - cx)**2 + (tracker.prev_cy - cy)**2) ** 0.5
            if dist < 80:
                return tracker
    new_id = f"person_{len(_person_trackers)}"
    t = PersonTracker(new_id)
    _person_trackers[len(_person_trackers)] = t
    return t

# Purge stale trackers (not updated in last 2 seconds)
_last_seen = {}

def _purge_stale():
    now = time.time()
    to_del = [k for k, t in _person_trackers.items()
              if _last_seen.get(k, 0) < now - 2.0]
    for k in to_del:
        del _person_trackers[k]
        _last_seen.pop(k, None)


# ─────────────────────────────────────────────────────────────
#  GEO HIERARCHY LOOKUP
# ─────────────────────────────────────────────────────────────
_REGIONS = ['Northern India','Southern India','Western India','Eastern India','Central India']
_CITIES = {
    'Northern India': ['Delhi','Chandigarh','Jaipur','Lucknow','Kanpur'],
    'Southern India': ['Chennai','Bengaluru','Hyderabad','Kochi','Madurai'],
    'Western India':  ['Mumbai','Ahmedabad','Pune','Surat','Nagpur'],
    'Eastern India':  ['Kolkata','Patna','Bhubaneswar','Guwahati','Ranchi'],
    'Central India':  ['Bhopal','Indore','Raipur','Jabalpur','Gwalior']
}

def _random_geo():
    region = random.choice(_REGIONS)
    city   = random.choice(_CITIES[region])
    cr     = f"CR-{random.randint(1,5)}"
    sensor = f"Camera-{random.randint(100,999)}"
    lat    = str(round(20.0 + random.uniform(-5, 10), 4))
    lng    = str(round(78.0 + random.uniform(-5, 10), 4))
    return region, city, cr, sensor, lat, lng


# ─────────────────────────────────────────────────────────────
#  FASTAPI LIFECYCLE
# ─────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_running_loop()
    threading.Thread(target=video_capture_thread, daemon=True).start()
    asyncio.create_task(dispatch_alarms())
    # Auto-start demo mode if env var is set
    if os.getenv("RAILDRISHTI_DEMO_MODE", "").lower() == "true":
        demo_mode.start_demo(alarm_queue, main_loop)

@app.websocket("/ws/alarms")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.add(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        clients.discard(websocket)

async def dispatch_alarms():
    while True:
        alarm_data = await alarm_queue.get()
        dead = set()
        for client in clients:
            try:
                await client.send_text(json.dumps(alarm_data))
            except Exception:
                dead.add(client)
        for c in dead:
            clients.discard(c)


# ─────────────────────────────────────────────────────────────
#  VIDEO + INFERENCE THREAD
# ─────────────────────────────────────────────────────────────
def video_capture_thread():
    global latest_frame
    cap = cv2.VideoCapture(0)
    last_alarm_time = 0

    while True:
        # ── Original initialisation with Reconnect ────────────────
        if not cap.isOpened():
            frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(frame, "CAMERA IN USE BY ANOTHER APP", (80, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            time.sleep(1)
            # Try to reconnect endlessly!
            cap.release()
            cap = cv2.VideoCapture(0)
        else:
            success, frame = cap.read()
            if not success:
                cap.release()
                cap = cv2.VideoCapture(0)
                continue

        # ── Skip inference during maintenance window ───────────
        maint = maintenance_mode
        if maint["active"] and time.time() < maint["until"]:
            cv2.putText(frame, "MAINTENANCE MODE — ALARMS SUPPRESSED",
                        (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            _encode_and_store(frame)
            time.sleep(0.04)
            continue

        # ── Inference ─────────────────────────────────────────
        if model is not None:
            results = list(model(frame, stream=True, verbose=False))
            tamper_trigger = False
            tamper_score   = 0.0
            _purge_stale()

            for r in results:
                # ── RESTORE ANNOTATED BOUNDING BOXES FOR USER ────
                frame = r.plot()
                
                boxes = r.boxes
                # Keypoints only available in pose mode
                kps_all = r.keypoints.data.cpu().numpy() if (MODEL_MODE == "pose" and r.keypoints is not None) else None

                for i, box in enumerate(boxes):
                    cls  = int(box.cls[0])
                    conf = float(box.conf[0])
                    name = model.names[cls]
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx = (x1 + x2) // 2
                    cy = (y1 + y2) // 2

                    # ── ANIMALS: draw + ignore for alarms ─────
                    if name in ['dog','cat','cow','horse','bird','sheep','elephant']:
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (0, 200, 0), 2)
                        cv2.putText(frame, f"Animal: {name}", (x1, y1-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1)
                        continue

                    # ── MAINTENANCE CREW marker (vehicles/equipment) ─
                    if name in ['car','truck','motorcycle','bus']:
                        cv2.rectangle(frame, (x1,y1), (x2,y2), (255,140,0), 2)
                        cv2.putText(frame, f"Vehicle: {name}", (x1, y1-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,140,0), 1)
                        continue

                    # ── PERSON: run through TBE ────────────────
                    if name == 'person' and conf > 0.4:
                        keypoints = None
                        if kps_all is not None and i < len(kps_all):
                            keypoints = kps_all[i]   # shape (17,3)

                        tracker = _find_or_create_tracker(cx, cy)
                        score = tracker.update(cx, cy, keypoints)

                        # Colour-code by suspicion level
                        if score >= 0.75:
                            colour = (0, 0, 255)    # RED — tampering
                            label  = f"TAMPER {int(score*100)}%"
                        elif score >= 0.45:
                            colour = (0, 140, 255)  # ORANGE — watch
                            label  = f"WATCH {int(score*100)}%"
                        else:
                            colour = (0, 220, 60)   # GREEN — passing
                            label  = f"PASS {int(score*100)}%"

                        cv2.rectangle(frame, (x1,y1), (x2,y2), colour, 2)
                        cv2.putText(frame, label, (x1, y1-8),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, colour, 2)

                        # Draw skeleton if available
                        if keypoints is not None:
                            SKELETON = [(0,1),(0,2),(1,3),(2,4),(5,6),
                                        (5,7),(7,9),(6,8),(8,10),
                                        (5,11),(6,12),(11,12),(11,13),
                                        (13,15),(12,14),(14,16)]
                            for a,b in SKELETON:
                                if a < len(keypoints) and b < len(keypoints):
                                    ka, kb = keypoints[a], keypoints[b]
                                    if ka[2] > 0.3 and kb[2] > 0.3:
                                        cv2.line(frame,
                                                 (int(ka[0]),int(ka[1])),
                                                 (int(kb[0]),int(kb[1])),
                                                 (180,180,60), 1)

                        if tracker.is_tamper_alarm():
                            tamper_trigger = True
                            tamper_score   = score
                            tracker.mark_alarmed()

            # ── Status overlay ─────────────────────────────────
            if tamper_trigger:
                cv2.putText(frame, f"TAMPERING DETECTED — CONF {int(tamper_score*100)}%",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
            else:
                cv2.putText(frame, "STATUS: CLEAR",
                            (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,220,60), 2)

            # ── Fire alarm if TBE triggered ────────────────────
            current_time = time.time()
            if tamper_trigger and current_time - last_alarm_time > 5 and main_loop is not None:
                try:
                    dist      = random.randint(1, 15)
                    direction = random.choice(["approaching","departing","approaching"])
                    level     = "critical" if (direction == "approaching" and dist <= 7) else "warning"
                    action    = "STOP TRAIN IMMEDIATELY" if level == "critical" else "Track Inspection Required"
                    msg       = (f"Track tampering detected — TBE Confidence: {int(tamper_score*100)}%. "
                                 f"Train {direction} at {dist}km. Action: {action}")

                    region, city, cr, sensor, lat, lng = _random_geo()
                    conf_pct = int(tamper_score * 100)

                    # ── DEDUPLICATION: reuse active incident if exists ──
                    existing_id = db.get_active_incident(sensor)
                    is_new = existing_id is None

                    if is_new:
                        incident_id = db.log_incident(msg, level, city, sensor,
                                                      current_time, region, city, cr, lat, lng)
                    else:
                        incident_id = existing_id
                        db.update_incident_confidence(incident_id, msg, conf_pct)

                    ws_payload = {
                        "id": incident_id,
                        "type": "alert" if is_new else "update",
                        "message": msg,
                        "timestamp": current_time, "level": level,
                        "zone": city, "region": region, "city": city,
                        "control_room": cr, "sensor": sensor,
                        "lat": lat, "lng": lng,
                        "tamper_confidence": conf_pct,
                        "status": "active"
                    }
                    main_loop.call_soon_threadsafe(alarm_queue.put_nowait, ws_payload)

                    # ── Fire autonomous responses only for NEW incidents ──
                    if is_new:
                        alarm_payload = {**ws_payload, "id": incident_id}
                        threading.Thread(
                            target=response_dispatcher.dispatch,
                            args=(alarm_payload,),
                            daemon=True
                        ).start()

                    last_alarm_time = current_time
                except Exception as e:
                    import traceback
                    print(f"ALARM CRASH: {e}")
                    traceback.print_exc()

        _encode_and_store(frame)
        time.sleep(0.01)


def _encode_and_store(frame):
    global latest_frame
    ret, buf = cv2.imencode('.jpg', frame)
    if ret:
        with latest_frame_condition:
            latest_frame = buf.tobytes()
            latest_frame_condition.notify_all()


# ─────────────────────────────────────────────────────────────
#  REST ENDPOINTS
# ─────────────────────────────────────────────────────────────
def frame_generator():
    while True:
        with latest_frame_condition:
            latest_frame_condition.wait(timeout=1.0)
            if latest_frame is None:
                continue
            frame_to_yield = latest_frame
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_to_yield + b'\r\n')
        time.sleep(0.03) # Yield thread briefly

@app.get("/video_feed")
def video_feed():
    return StreamingResponse(frame_generator(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/api/system_health")
def system_health():
    return {
        "status": "online",
        "vision_model": f"YOLOv8-Pose TBE ({MODEL_MODE})",
        "active_cameras": 1,
        "maintenance_mode": maintenance_mode["active"] and time.time() < maintenance_mode["until"]
    }

class AckRequest(BaseModel):
    comment: str

class MaintenanceRequest(BaseModel):
    duration_minutes: int   # how long to suppress alarms

@app.post("/api/maintenance/enable")
def enable_maintenance(req: MaintenanceRequest):
    maintenance_mode["active"] = True
    maintenance_mode["until"]  = time.time() + req.duration_minutes * 60
    return {"status": "maintenance_active", "until": maintenance_mode["until"]}

@app.post("/api/maintenance/disable")
def disable_maintenance():
    maintenance_mode["active"] = False
    maintenance_mode["until"]  = 0
    return {"status": "maintenance_disabled"}

@app.post("/api/demo/start")
async def start_demo_mode():
    demo_mode.start_demo(alarm_queue, main_loop)
    return {"status": "demo_started"}

@app.post("/api/demo/stop")
async def stop_demo_mode():
    demo_mode.stop_demo()
    return {"status": "demo_stopped"}

@app.delete("/api/incidents/clear")
async def clear_all_incidents():
    db.clear_incidents()
    return {"status": "cleared"}

@app.get("/api/incidents/{incident_id}/response_log")
def get_incident_response_log(incident_id: int):
    return db.get_response_log(incident_id)

@app.get("/api/incidents")
def get_incidents(zone: str = None):
    return db.get_incidents(zone)

@app.post("/api/incidents/{incident_id}/acknowledge")
def ack_incident(incident_id: int, req: AckRequest):
    db.acknowledge_incident(incident_id, req.comment)
    return {"status": "success"}

if os.path.exists("frontend/dist/assets"):
    app.mount("/assets", StaticFiles(directory="frontend/dist/assets"), name="assets")

    @app.get("/{catchall:path}")
    async def serve_frontend(catchall: str = ""):
        if catchall == "" or not catchall.startswith("api/"):
            return FileResponse("frontend/dist/index.html")
