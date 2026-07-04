# Auto-generated Prostate Cancer CADx Skill
# ID: 26 | Name: download_panda

import sys
from pathlib import Path

def main():
    print("--- Running skill 26: download_panda ---")

    import subprocess
    subprocess.run(["python", "scripts/download_data.py"])

    print("--- Skill 26 completed successfully ---")

if __name__ == "__main__":
    main()
