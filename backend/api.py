import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import pandas as pd
from lib.db import db
from lib.gpu import gpu_monitor
from lib.config import config

app = FastAPI(title="Prostate Cancer CADx Supervisor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/status")
def get_status():
    skills = db.get_all_skills()
    done = [s for s in skills if s["status"] == "done"]
    pending = [s for s in skills if s["status"] == "pending"]
    blocked = [s for s in skills if s["status"] == "blocked"]
    running = [s for s in skills if s["status"] == "running"]
    
    return {
        "status": "active",
        "total_skills": len(skills),
        "done_count": len(done),
        "pending_count": len(pending),
        "blocked_count": len(blocked),
        "running_count": len(running),
        "completion_rate": len(done) / len(skills) if skills else 0
    }

@app.get("/metrics")
def get_metrics():
    latest = db.get_latest_metrics()
    return latest if latest else {"message": "No metrics logged yet."}

@app.get("/skills")
def get_skills():
    return db.get_all_skills()

@app.get("/gpu")
def get_gpu():
    csv_path = Path("logs/gpu_util.csv")
    if csv_path.exists():
        try:
            df = pd.read_csv(csv_path)
            tail = df.tail(10).to_dict(orient="records")
            return {
                "latest_readings": tail,
                "peak_util": float(df["gpu_util_pct"].max()) if "gpu_util_pct" in df else 0.0,
                "avg_util": float(df["gpu_util_pct"].mean()) if "gpu_util_pct" in df else 0.0
            }
        except Exception as e:
            return {"error": str(e)}
    return {"message": "GPU logs not generated yet."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8600)
