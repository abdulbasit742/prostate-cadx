# docs/REPRODUCE.md — Exact Reproduction Commands

## Environment
```
Python 3.11, PyTorch 2.6.0+cu124, sklearn 1.9.0
GPU: NVIDIA RTX A6000 48GB
OS: Windows 11
```

## 1. Data Setup
```powershell
# Activate venv
.\venv\Scripts\activate
# Download SICAPv2 from Zenodo 14178894 (requires ZENODO_ACCESS_TOKEN in config/.env)
python scripts/download_data.py
```

## 2. Training (seed=42)
```powershell
python scripts/train.py --lr 0.0002 --weight_decay 0.05 --backbone resnet50
# Checkpoint saved to: storage/checkpoints/best_model.pt
```

## 3. Corrected Evaluation (Audit-verified, 3 seeds)
```powershell
python scripts/full_audit_eval.py
# Outputs: AUDIT.md, RESULTS.md, docs/assets/*.png
```

## 4. Simulated Domain Shift
```powershell
python scripts/simulated_domain_shift.py
```

## 5. Label Audit
```powershell
python scripts/label_audit.py
```

## Key Fixes Applied
- **Split leakage**: `drop_duplicates()` → `groupby("slide_id")["slide_isup"].max()`
- **Seeds**: All evaluations use `random_state=42` (or 42/43/44 for 3-run SD estimates)
- **QWK implementation**: `sklearn.metrics.cohen_kappa_score(weights='quadratic')`

## Honest Limitations
- SICAPv2 covers ISUP 0/1/4/5 only (no ISUP 2/3)
- Domain shift is SIMULATED, not a real TCGA external validation
- Confidence (softmax max-prob) is uncalibrated; temperature scaling not yet applied
- Loss ablation (CORAL, Soft-QWK) metrics need real training runs
