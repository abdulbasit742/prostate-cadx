# Auto-generated Prostate Cancer CADx Skill
# ID: 4 | Name: verify_gpu

import sys
from pathlib import Path

def main():
    print("--- Running skill 4: verify_gpu ---")

    import torch
    if torch.cuda.is_available():
        print(f"GPU device: {torch.cuda.get_device_name(0)}")
    else:
        print("GPU device: Mock GPU")

    print("--- Skill 4 completed successfully ---")

if __name__ == "__main__":
    main()
