import os
import sys
import subprocess
from pathlib import Path

# Inject project root into sys.path
project_root = str(Path(__file__).resolve().parent.parent)
sys.path.insert(0, project_root)

from lib.logging_setup import logger
from lib.db import db

def run_step(name, cmd):
    logger.info(f"=== RUNNING STEP: {name} ===")
    logger.info(f"Command: {' '.join(cmd)}")
    
    python_path = Path("venv/Scripts/python.exe")
    if not python_path.exists():
        python_path = Path(sys.executable)
        
    env = os.environ.copy()
    env["PYTHONPATH"] = project_root
        
    full_cmd = [str(python_path)] + cmd
    res = subprocess.run(full_cmd, env=env)
    if res.returncode != 0:
        logger.error(f"Step {name} failed with return code {res.returncode}")
        sys.exit(res.returncode)
    logger.info(f"=== STEP {name} COMPLETED SUCCESSFUL ===\n")

def main():
    logger.info("Starting Prostate Cancer CADx Smoke Harness Verification...")
    
    # 1. Initialize registry
    run_step("Registry Init", ["scripts/init_registry.py"])
    
    # 2. Seed registry into DB
    db.log_event("INFO", "Running initial seeds setup.")
    import json
    seed_path = Path("loop/skills_seed.json")
    if seed_path.exists():
        with open(seed_path, "r") as f:
            seeds = json.load(f)
            for skill in seeds:
                db.register_skill(
                    id=skill["id"],
                    name=skill["name"],
                    group_name=skill["group"],
                    status="pending",
                    deps=skill["deps"]
                )
        db.log_event("INFO", f"Seeded {len(seeds)} skills into SQLite database.")

    # 3. Download data (creates synthetic files)
    run_step("Data Download", ["scripts/download_data.py"])
    
    # 4. Tiling
    run_step("Slide Tiling", ["scripts/tile_wsi.py"])
    
    # 5. Training in smoke mode
    run_step("Model Training", ["scripts/train.py", "--smoke"])
    
    # 6. Evaluation
    run_step("Model Evaluation", ["scripts/evaluate.py"])
    
    # 7. Grad-CAM demo
    run_step("Grad-CAM Demo", ["scripts/gradcam_demo.py"])
    
    logger.info("PROSTATE CANCER CADX SMOKE PIPELINE VERIFIED SUCCESSFULLY!")

if __name__ == "__main__":
    main()
