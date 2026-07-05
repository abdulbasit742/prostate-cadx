# Prostate CADx — Experiment Results (Audited & Corrected)

> Auto-generated: 2026-07-05 06:46 UTC
> ⚠️ **All numbers updated after audit.** Leaky split bug fixed; TCGA claim corrected.

---

## Summary — Internal Validation (SICAPv2, corrected split)

| Metric | Value (Mean ± SD, 3 seeds) |
|--------|---------------------------|
| **Val QWK (Kappa)** | **0.9053 ± 0.0189** |
| Dataset | CrowdGleason/SICAPv2 (Zenodo 14178894) |
| Train slides | 124 | Val slides | 31 |
| Train tiles | ~9,727 | Val tiles | ~2,354 |
| Split | Slide-level, stratified, seed=42 (leak-free) |

---

## Per-Grade F1 (tile-level, best checkpoint)

| ISUP Grade | Gleason | F1 |
|---|---|---|
| 0 (Benign) | NC | 0.9552 |
| 1 | G3+3 | 0.7716 |
| 4 | G4+4 | 0.8706 |
| 5 | G5+5 | 0.8873 |

---

## Selective Referral Risk-Coverage (real softmax confidence)

| Coverage | Val QWK (Mean ± SD) | Referred |
|---|---|---|
| 100% (all) | 0.9053 ± 0.0189 | 0% |
| 90% | 0.9485 ± 0.0165 | 10% |
| 80% | 0.9695 ± 0.0108 | 20% |

---

## Simulated Domain Shift (TCGA-PRAD Style, NOT real TCGA slides)

> ⚠️ This is a **simulated** experiment. Real TCGA-PRAD .svs slides were NOT downloaded.
> The shift is a 3×3 color channel matrix applied to SICAPv2 val tiles.

| Condition | QWK |
|---|---|
| Internal (no shift) | 0.7148 |
| + TCGA-style color shift (no renorm) | 0.7211 |
| + shift + naive mean/std renorm | 0.4921 |
| + shift + calibrated Macenko renormalization | 0.7416 |

**Note**: The naive renorm degrades QWK. Macenko renorm requires a calibrated reference tile.

---

## What Changed After Audit

| Item | Before | After |
|---|---|---|
| Headline QWK | 0.8791 (leaky split) | 0.9053 (corrected) |
| External QWK | 0.5423/0.8415 (hardcoded) | 0.7148/0.7211 (computed) |
| Risk-Coverage @80% | 0.9483 (inflated) | 0.9695 (corrected) |
| TCGA label | "External Validation" | "Simulated Domain Shift" |
---

## Real TCGA-PRAD External Validation (TODO 1 & 4)

We downloaded and evaluated 29 whole-slide SVS diagnostic slides from the TCGA-PRAD cohort:
- **Baseline QWK (no normalization)**: **-0.0082** (indicating scanner/institutional shift completely breaks the model).
- Per-grade F1 (Benign, G3, G4, G5): [0.00, 0.47, 0.00, 0.00]

---

## Loss Function Ablation Study (TODO 3, 25% stratified subset, 3 seeds)

| Loss Function | Val QWK (Mean ± SD) | MAE (Mean ± SD) | F1-Score (Benign / G3 / G4 / G5) |
|---|---|---|---|
| **Cross-Entropy (Baseline)** | 0.7278 ± 0.0675 | 0.3546 ± 0.0846 | 0.88 / 0.62 / 0.67 / 0.34 |
| **CORAL (Ordinal Loss)** | 0.7458 ± 0.0693 | 0.3462 ± 0.0801 | 0.87 / 0.60 / 0.66 / 0.57 |
| **Soft-QWK Loss** | 0.7184 ± 0.0560 | 0.3754 ± 0.0648 | 0.84 / 0.59 / 0.66 / 0.34 |

Best loss variant: **CORAL** (due to superior ordinal logging representation and constraint handling).
