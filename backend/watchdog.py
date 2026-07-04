import time
import subprocess
import sys
from pathlib import Path
from lib.db import db
from lib.logging_setup import logger

def is_daemon_running():
    # Simple check for daemon.py process
    try:
        if sys.platform == "win32":
            cmd = "wmic process get CommandLine"
            output = subprocess.check_output(cmd, shell=True, text=True)
            return "backend/daemon.py" in output or "backend\\daemon.py" in output
        else:
            cmd = "ps aux | grep daemon.py"
            output = subprocess.check_output(cmd, shell=True, text=True)
            return "daemon.py" in output
    except Exception:
        return False

def start_daemon():
    venv_python = Path("venv/Scripts/python.exe")
    if not venv_python.exists():
        venv_python = Path(sys.executable)
        
    db.log_event("WARNING", "Watchdog: daemon process not detected. Restarting...")
    
    # Start as background process
    subprocess.Popen(
        [str(venv_python), "backend/daemon.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True
    )
    
    db.log_event("INFO", "Watchdog: successfully restarted backend/daemon.py.")

def main():
    logger.info("Watchdog started. Monitoring daemon...")
    while True:
        if not is_daemon_running():
            start_daemon()
        time.sleep(15)

if __name__ == "__main__":
    main()
