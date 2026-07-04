# Auto-generated Prostate Cancer CADx Skill
# ID: 3 | Name: install_cuda_torch

import sys
from pathlib import Path

def main():
    print("--- Running skill 3: install_cuda_torch ---")

    import torch
    print(f"CUDA available: {torch.cuda.is_available()}")

    print("--- Skill 3 completed successfully ---")

if __name__ == "__main__":
    main()
