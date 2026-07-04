import os
import sys
import time
import subprocess
import json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datetime import datetime
from lib.db import db
from lib.logging_setup import logger
from lib.config import config
from loop.engine import LoopEngine

class DaemonSupervisor:
    def __init__(self):
        self.venv_python = Path("venv/Scripts/python.exe")
        if not self.venv_python.exists():
            self.venv_python = Path(sys.executable)
        self.skills_dir = Path("skills")
        self.logs_dir = Path("logs/skills")
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        self.commit_queue = []
        self.engine = LoopEngine()

    def get_ready_skill(self):
        return self.engine.get_next_ready_skill()

    def execute_skill(self, skill) -> bool:
        sid = skill["id"]
        sname = skill["name"]
        
        skill_file = self.skills_dir / f"skill_{sid}_{sname}.py"
        if not skill_file.exists():
            db.update_skill_status(sid, "blocked", f"File {skill_file.name} does not exist.")
            return False

        log_file = self.logs_dir / f"skill_{sid}_{sname}.log"
        db.update_skill_status(sid, "running")
        run_id = db.record_run_start(sid, str(log_file))
        
        db.log_event("INFO", f"Daemon executing skill {sid} ({sname})...")
        
        try:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parent.parent)
            
            with open(log_file, "w") as f:
                res = subprocess.run(
                    [str(self.venv_python), str(skill_file), "--smoke"],
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=300,
                    env=env
                )
            
            ok = (res.returncode == 0)
            db.record_run_end(run_id, ok)
            
            if ok:
                db.update_skill_status(sid, "done")
                db.log_event("INFO", f"Skill {sid} ({sname}) completed successfully.")
                self.queue_commit(sname)
                return True
            else:
                db.update_skill_status(sid, "blocked", f"Execution failed with code {res.returncode}")
                db.log_event("ERROR", f"Skill {sid} ({sname}) failed execution.")
                return False
        except subprocess.TimeoutExpired:
            db.record_run_end(run_id, False)
            db.update_skill_status(sid, "blocked", "Execution timeout.")
            db.log_event("ERROR", f"Skill {sid} ({sname}) timed out.")
            return False
        except Exception as e:
            db.record_run_end(run_id, False)
            db.update_skill_status(sid, "blocked", str(e))
            db.log_event("ERROR", f"Skill {sid} ({sname}) error: {e}")
            return False

    def queue_commit(self, skill_name):
        """
        Git commit helper. Commits changes to keep repo tracked.
        """
        self.commit_queue.append(skill_name)
        self.process_commit_queue()

    def process_commit_queue(self):
        try:
            # Check if git is initialized
            res = subprocess.run(["git", "status"], capture_output=True)
            if res.returncode != 0:
                return # Not a git repo or git missing
                
            while self.commit_queue:
                sname = self.commit_queue.pop(0)
                subprocess.run(["git", "add", "."])
                subprocess.run(["git", "commit", "-m", f"feat(skill): implement {sname}"])
                db.log_event("INFO", f"Git committed skill: {sname}")
        except Exception as e:
            logger.warning(f"Git commit failed: {e}")

    def run_forever(self):
        db.log_event("INFO", "Prostate Cancer CADx Daemon started.")
        
        while True:
            skill = self.get_ready_skill()
            if skill:
                # Run the skill with retry policy (retry up to 2 times on failure)
                retries = 2
                success = False
                while retries >= 0:
                    success = self.execute_skill(skill)
                    if success:
                        break
                    else:
                        retries -= 1
                        if retries >= 0:
                            db.log_event("WARNING", f"Retrying skill {skill['name']} ({retries} retries left)...")
                            time.sleep(2)
                            
                if not success:
                    db.log_event("ERROR", f"Skill {skill['name']} blocked after retries.")
            else:
                # IMPROVE MODE or Idle loop
                all_skills = db.get_all_skills()
                pending = [s for s in all_skills if s["status"] == "pending"]
                blocked = [s for s in all_skills if s["status"] == "blocked"]
                
                if not pending and not blocked:
                    # Enters IMPROVE MODE - active retraining loop
                    db.log_event("INFO", "All skills completed. Enters IMPROVE MODE. Launching full training cycle to maximize GPU performance...")
                    try:
                        # Run full training script to train model and evaluate metrics
                        subprocess.run(
                            [str(self.venv_python), "scripts/train.py"],
                            check=True
                        )
                        db.log_event("INFO", "Full training cycle completed successfully. Validation metrics logged.")
                    except Exception as e:
                        db.log_event("ERROR", f"Retraining cycle failed: {e}")
                    
                    time.sleep(30)
                else:
                    # Some skills blocked, or waiting for dependencies
                    time.sleep(10)

if __name__ == "__main__":
    daemon = DaemonSupervisor()
    daemon.run_forever()
