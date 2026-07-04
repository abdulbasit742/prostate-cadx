import subprocess
import csv
import time
import os
import threading
from pathlib import Path
from lib.logging_setup import logger

CSV_PATH = Path("logs/gpu_util.csv")

class GPUMonitor:
    def __init__(self, csv_path=CSV_PATH):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        self.running = False
        self.monitor_thread = None
        self._init_csv()

    def _init_csv(self):
        if not self.csv_path.exists():
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["timestamp", "gpu_util_pct", "memory_used_mb", "memory_total_mb"])

    def get_gpu_status(self):
        try:
            # Query nvidia-smi for GPU utilization and memory usage
            cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
            output = subprocess.check_output(cmd, text=True).strip()
            util, mem_used, mem_total = map(float, output.split(","))
            return util, mem_used, mem_total
        except Exception as e:
            # Fallback if nvidia-smi fails or is unavailable
            return 0.0, 0.0, 0.0

    def _monitor_loop(self):
        self.running = True
        while self.running:
            util, used, total = self.get_gpu_status()
            try:
                with open(self.csv_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([time.strftime("%Y-%m-%d %H:%M:%S"), util, used, total])
            except Exception as e:
                pass
            time.sleep(30)

    def start(self):
        if self.monitor_thread is None or not self.monitor_thread.is_alive():
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            logger.info("GPU Monitor started.")

    def stop(self):
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=2)
            logger.info("GPU Monitor stopped.")

# Global monitor instance
gpu_monitor = GPUMonitor()
