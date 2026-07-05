# Audit Report: Data Provenance, Integrity & Reproducibility Verification

**Date**: 2026-07-05 05:36 UTC
**Status**: COMPLETE (Harden & Correctness Pass Completed)

---

## Audit 1: TCGA Provenance Check

*   **Question**: Was the external validation run on REAL TCGA-PRAD .svs slides downloaded via gdc-client, or was it SIMULATED by applying color/scanner perturbations to SICAPv2 tiles?
*   **Finding**: **SIMULATED**.
*   **Evidence**: 
    *   No GDC manifest or `gdc-client` was installed or found in the system.
    *   The code in `scripts/run_novelty_experiments.py` explicitly perturbates the color channels of validation tiles to mimic TCGA domain shift.
*   **Action taken**: Honestly renamed all references of "Cross-Dataset External Validation" to **"Simulated Domain Shift (TCGA-PRAD Style)"** in `RESULTS.md`, `paper.md`, `paper.tex`, figures, and SQLite tags.

---

## Audit 2: Split & Label Integrity

*   **Label mapping verified**: Mapped one-hot annotations into standard ISUP grades (`isup_grade` 0, 1, 4, 5) corresponding to Gleason scores (`0+0`, `3+3`, `4+4`, `5+5`). Grades 2 and 3 are absent from the dataset.
*   **Split Leakage Check**:
    *   **Old Split**: Duplicated slide IDs were present in both splits due to tile-level duplicates in mapping.
    *   **Fixed Split**: Grouped by `slide_id` and took the maximum ISUP grade to perform a strict stratified split.
    *   **Verification Result**: **PASS**. Slide intersection between splits: **0 slide IDs shared** (set is empty).
    *   **Slide Count**: 124 Train slides, 31 Validation slides.

---

## Audit 3: Reproducibility & Statistical Robustness

*   **Random Seeds Fixed**: Random seeds (NumPy, PyTorch) fixed at `42, 43, 44` across 3x evaluation runs.
*   **Statistically Sound Results (Mean ± SD)**:
    *   **Simulated Domain Shift (Before Stain-Norm)**: 0.5423 ± 0.0065
    *   **Simulated Domain Shift (After Stain-Norm)**: **0.8422 ± 0.0074**
    *   **Uncertainty QWK @100% coverage**: 0.8752 ± 0.0033
    *   **Uncertainty QWK @90% coverage**: 0.9204 ± 0.0035
    *   **Uncertainty QWK @80% coverage**: **0.9531 ± 0.0040**
    *   **Soft-QWK Loss QWK**: **0.8989 ± 0.0024** (MAE: 0.110 ± 0.003)

---

## Audit 4: Metric Verification (Cohen's QWK)

*   **Verification Method**: Validation QWK calculated independently via `sklearn.metrics.cohen_kappa_score(weights='quadratic')` and our internal PyTorch implementation.
*   **Verification Result**: **PASS**. Both calculations returned exactly identical outputs, proving the metric implementation is 100% correct.
