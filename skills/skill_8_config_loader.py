# Auto-generated Prostate Cancer CADx Skill
# ID: 8 | Name: config_loader

import sys
from pathlib import Path

def main():
    print("--- Running skill 8: config_loader ---")

    from lib.config import config
    print(f"Config loaded. Tile size: {config.get('data.tile_size')}")

    print("--- Skill 8 completed successfully ---")

if __name__ == "__main__":
    main()
