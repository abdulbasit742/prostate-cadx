import os
import yaml
from pathlib import Path

class Config:
    def __init__(self, config_path="config/config.yaml", env_path="config/.env"):
        self.config_path = Path(config_path)
        self.env_path = Path(env_path)
        self.cfg = {}
        self.load_config()
        self.load_env()

    def load_config(self):
        if self.config_path.exists():
            with open(self.config_path, "r") as f:
                self.cfg = yaml.safe_load(f)
        else:
            self.cfg = {}

    def load_env(self):
        if self.env_path.exists():
            with open(self.env_path, "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):
                        key, val = line.split("=", 1)
                        os.environ[key.strip()] = val.strip().strip('"').strip("'")

    def get(self, key, default=None):
        keys = key.split(".")
        val = self.cfg
        for k in keys:
            if isinstance(val, dict) and k in val:
                val = val[k]
            else:
                return default
        return val

# Global configuration instance
config = Config()
