# Railway AI Safety System - Version 3 🚆

Welcome to **Version 3** of the Railway AI Safety System! This directory contains the most advanced and comprehensive iteration of the project, building upon the foundations of Version 1 (Streamlit prototype) and Version 2 (Full-stack React/FastAPI).

## What's New in Version 3?

Version 3 significantly enhances the AI capabilities and structural robustness of the system:

- **Advanced ML Pipeline:** Incorporates a robust machine learning layer with comprehensive statistics and cross-validation output logging (`ml_stats_output.txt`, `ml_stats_cv_output.txt`).
- **Behavior Engine:** A newly implemented `behavior_engine.py` for advanced risk assessment and contextual analysis of detected anomalies.
- **Enhanced YOLO Integration:** Further refined YOLO-based object detection (`yolo_detector.py`) with support for pose estimation (`yolov8n-pose.pt`) and medium/nano models (`yolov8m.pt`, `yolov8n.pt`).
- **Feature Extraction:** Advanced feature extraction logic for more granular data analysis.
- **Comprehensive Datasets:** Includes `datasets` and `data` directories containing the master training data for robust model performance.
- **Production-Ready Stack:** Continues to utilize the high-performance FastAPI backend and React + Vite frontend for a seamless user experience.

## Directory Structure

- `frontend/`: The React + Vite user interface application.
- `server/`: The FastAPI backend server and API routes.
- `src/`: The core machine learning and detection logic (`detection/`, `features/`, `models/`, `pipeline/`, etc.).
- `datasets/` & `data/`: The training data and resources used by the AI models.
- `models/`: The serialized models and training outputs (including metrics graphs and YOLO stats).

## Getting Started

1. **Install Dependencies:**
   Ensure you have installed the required packages listed in `requirements.txt`.

2. **Run the Backend (Server):**
   Navigate to the `server/` directory and start the FastAPI application.

3. **Run the Frontend:**
   Navigate to the `frontend/` directory, install dependencies (`npm install`), and start the Vite development server (`npm run dev`).

*Developed by the CodingKDhurandar Team for the Hack4Delhi Hackathon.*
