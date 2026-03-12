# Railway AI System (Version 2) 🚀

Version 2 represents the production-ready architecture of the Railway AI Safety System. We migrated from Streamlit to a full **FastAPI and React** stack, integrating YOLOv8 for powerful machine learning capabilities.

## React Dashboard Preview



## Architecture Highlights
- **FastAPI Backend**: Robust, asynchronous REST APIs managing sensors, alerts, and AI inference requests. SQLite database for persistence.
- **React + Vite Frontend**: A modern, snappy dashboard mimicking a real Railway Control Room.
- **YOLOv8 AI Models**: Custom trained `yolov8n` and `yolov8n-pose` models to detect structural track damage and human intrusion on CCTV feeds.

## How to Run

You will need two terminal windows to run both the backend and frontend simultaneously.

### 1. Start the FastAPI Backend
```bash
cd railway_ai_system
pip install -r requirements.txt
uvicorn api.main:app --reload --port 8000
```

### 2. Start the React Frontend
```bash
cd railway_ai_system/frontend
npm install
npm run dev
```
Then, open the URL provided by Vite (usually `http://localhost:5173`) in your browser to access the Control Room.