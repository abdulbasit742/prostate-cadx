import sqlite3
import os
import json
from datetime import datetime
from pathlib import Path
from lib.logging_setup import logger

DB_PATH = Path("db/cadx.db")

class Database:
    def __init__(self, db_path=DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.init_db()

    def get_connection(self):
        return sqlite3.connect(self.db_path)

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            # 1. skills table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS skills (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    group_name TEXT NOT NULL,
                    status TEXT NOT NULL, -- pending, running, done, blocked
                    deps TEXT NOT NULL, -- JSON string array
                    blocked_reason TEXT
                )
            """)
            # 2. runs table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    skill_id INTEGER NOT NULL,
                    started_at TEXT NOT NULL,
                    ended_at TEXT,
                    ok INTEGER, -- 0 or 1
                    log_path TEXT,
                    gpu_avg REAL,
                    gpu_peak REAL
                )
            """)
            # 3. metrics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    kappa REAL,
                    val_loss REAL,
                    batch_size INTEGER,
                    epoch INTEGER,
                    checkpoint_path TEXT
                )
            """)
            # 4. events table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts TEXT NOT NULL,
                    level TEXT NOT NULL,
                    message TEXT NOT NULL
                )
            """)
            conn.commit()

    def log_event(self, level: str, message: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO events (ts, level, message) VALUES (?, ?, ?)",
                (datetime.utcnow().isoformat(), level, message)
            )
            conn.commit()
        # Also print/log to standard logger
        if level.upper() == "INFO":
            logger.info(message)
        elif level.upper() == "WARNING":
            logger.warning(message)
        elif level.upper() == "ERROR":
            logger.error(message)

    def register_skill(self, id: int, name: str, group_name: str, status: str, deps: list):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT OR REPLACE INTO skills (id, name, group_name, status, deps) VALUES (?, ?, ?, ?, ?)",
                (id, name, group_name, status, json.dumps(deps))
            )
            conn.commit()

    def update_skill_status(self, skill_id: int, status: str, blocked_reason: str = None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE skills SET status = ?, blocked_reason = ? WHERE id = ?",
                (status, blocked_reason, skill_id)
            )
            conn.commit()
            self.log_event("INFO", f"Skill ID {skill_id} status updated to {status}")

    def get_all_skills(self):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM skills")
            return [dict(row) for row in cursor.fetchall()]

    def get_pending_skills(self):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM skills WHERE status = 'pending'")
            return [dict(row) for row in cursor.fetchall()]

    def record_run_start(self, skill_id: int, log_path: str) -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO runs (skill_id, started_at, log_path) VALUES (?, ?, ?)",
                (skill_id, datetime.utcnow().isoformat(), log_path)
            )
            run_id = cursor.lastrowid
            conn.commit()
            return run_id

    def record_run_end(self, run_id: int, ok: bool, gpu_avg: float = None, gpu_peak: float = None):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE runs SET ended_at = ?, ok = ?, gpu_avg = ?, gpu_peak = ? WHERE id = ?",
                (datetime.utcnow().isoformat(), 1 if ok else 0, gpu_avg, gpu_peak, run_id)
            )
            conn.commit()

    def add_metrics(self, kappa: float, val_loss: float, batch_size: int, epoch: int, checkpoint_path: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO metrics (ts, kappa, val_loss, batch_size, epoch, checkpoint_path) VALUES (?, ?, ?, ?, ?, ?)",
                (datetime.utcnow().isoformat(), kappa, val_loss, batch_size, epoch, checkpoint_path)
            )
            conn.commit()
            self.log_event("INFO", f"Logged metrics - Epoch {epoch}: Kappa={kappa:.4f}, ValLoss={val_loss:.4f}")

    def get_latest_metrics(self):
        with self.get_connection() as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM metrics ORDER BY id DESC LIMIT 1")
            row = cursor.fetchone()
            return dict(row) if row else None

# Global DB instance
db = Database()
