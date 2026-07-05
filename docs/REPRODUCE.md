# Reproducibility Guide: Prostate Cancer CADx

This guide outlines the commands and seeds required to reproduce all findings, figures, and tables reported in the paper.

## 1. Setup Environment
Ensure that the virtual environment is active and dependencies are fully installed:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Running Data Ingestion & Tiling
To verify the dataset mapping and recreate the manifest:
```bash
python scripts/download_data.py
python scripts/tile_wsi.py
```

## 3. Train the Baseline Model
To run training from scratch with standard hyperparameters (seed 42):
```bash
python scripts/train.py --lr 0.0002 --weight_decay 0.05 --backbone resnet50
```

## 4. Run Audited Experiments & Regenerate Plots
To execute the 4 novelty evaluations (simulated domain shift, saliency overlap, selective referral, loss ablation) across 3x random seeds and regenerate all publication-ready figures:
```bash
python scripts/run_reproducible_audit.py
```

All figures will be output to `docs/assets/` and tag fields in `docs/paper/paper.md` will be updated with the statistical values (Mean ± SD).
