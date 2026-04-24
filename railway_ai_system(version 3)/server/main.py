import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pathlib import Path

from server.database import engine
from server import models

# Automatically build local database structure
models.Base.metadata.create_all(bind=engine)

# Placeholder route imports
from server.routes import analyze_routes, results_routes, config_routes, alert_routes, history_routes

# Load configuration
config_path = Path(__file__).parent.parent / "config.yaml"
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

app = FastAPI(
    title=config["project"]["name"],
    version=config["project"]["version"],
    description="Railway Tampering Detection Batch Processing API"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=config["server"]["cors_origins"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve raw frames dynamically for the frontend Gallery UI
app.mount("/api/data", StaticFiles(directory="data"), name="data")

# Include routers
app.include_router(analyze_routes.router, prefix="/api/analyze", tags=["Analyze"])
app.include_router(results_routes.router, prefix="/api/results", tags=["Results"])
app.include_router(config_routes.router, prefix="/api/config", tags=["Config"])
app.include_router(alert_routes.router, prefix="/api/alerts", tags=["Alerts"])
app.include_router(history_routes.router, prefix="/api/history", tags=["History"])

@app.get("/")
def read_root():
    return {"status": "ok", "service": "RailGuard API v1.0", "mode": "batch-processing"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.main:app", host=config["server"]["host"], port=config["server"]["port"], reload=True)
