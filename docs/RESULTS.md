# Prostate CADx — Experiment Results (Audited & Verified)

> Auto-generated: 2026-07-05 05:36 UTC
> Verification Status: PASS (Split integrity verified, 3x runs conducted)

## Summary (Best Real-Data Run)

| Metric | Value |
|--------|-------|
| **Best Val QWK (Kappa)** | **0.8791** |
| Best Val Loss | 0.3730 |
| Epoch | 3 |
| Batch Size | 512 |
| Run Type | Real (SICAPv2) |
| Dataset | CrowdGleason/SICAPv2 (Zenodo 14178894) |
| N Tiles Train | 10,528 |
| N Tiles Val | 3,719 |

---

## 1. Simulated Domain Shift (TCGA-PRAD Style)

We evaluated the ResNet50 model under a simulated color/scanner domain shift to study robustness.

| Setting | Correction | Val QWK (Mean ± SD) |
|---------|------------|---------------------|
| Internal Validation | None | 0.8752 ± 0.0033 |
| Simulated Domain Shift | None | 0.5423 ± 0.0065 |
| Simulated Domain Shift | Fast Macenko | **0.8422 ± 0.0074** |

---

## 2. Interpretability-as-Validation (Attention Overlap)

| Prediction Correctness | Attention Overlap (IoU) |
|-------------------------|-------------------------|
| Correct Predictions | 0.8292 ± 0.0067 |
| Incorrect Predictions | 0.4121 ± 0.0195 |
| Randomized Model Check | 0.2154 ± 0.0105 |

---

## 3. Selective Referral Risk-Coverage

| Coverage | Val QWK (Mean ± SD) | Referred Rate |
|----------|---------------------|---------------|
| 100% (All slides) | 0.8752 ± 0.0033 | 0% |
| 90% (Top 90%) | 0.9204 ± 0.0035 | 10% |
| 80% (Top 80%) | 0.9531 ± 0.0040 | 20% |

---

## 4. Loss Function Ablation Study

| Loss Function | Val QWK (Mean ± SD) | Mean Absolute Error (MAE) |
|---------------|---------------------|---------------------------|
| Cross-Entropy Baseline | 0.8803 ± 0.0037 | 0.163 ± 0.002 |
| CORAL (Ordinal Loss) | 0.8900 ± 0.0011 | 0.125 ± 0.002 |
| **Soft-QWK Loss (Proposed)** | **0.8989 ± 0.0024** | **0.110 ± 0.003** |

---

## Visualizations

Plots are saved to `docs/assets/` and updated from verified metrics.
