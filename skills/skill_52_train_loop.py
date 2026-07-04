# Auto-generated Prostate Cancer CADx Skill
# ID: 52 | Name: train_loop

import sys
from pathlib import Path

def main():
    print("--- Running skill 52: train_loop ---")

    import subprocess
    subprocess.run(["python", "scripts/train.py", "--smoke"])

    print("--- Skill 52 completed successfully ---")

if __name__ == "__main__":
    main()
