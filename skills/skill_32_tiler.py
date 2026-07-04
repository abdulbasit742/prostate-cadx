# Auto-generated Prostate Cancer CADx Skill
# ID: 32 | Name: tiler

import sys
from pathlib import Path

def main():
    print("--- Running skill 32: tiler ---")

    import subprocess
    subprocess.run(["python", "scripts/tile_wsi.py"])

    print("--- Skill 32 completed successfully ---")

if __name__ == "__main__":
    main()
