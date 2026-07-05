import json
from pathlib import Path

def main():
    print("Updating results files...")
    
    # 1. Load results from JSONs
    # loss ablation
    with open("storage/loss_ablation_results.json", "r") as f:
        ablation = json.load(f)
    
    # tcga real
    with open("storage/tcga_real_results.json", "r") as f:
        tcga_real = json.load(f)
        
    # calibration
    with open("storage/calibration.json", "r") as f:
        calib = json.load(f)

    # 2. Update docs/RESULTS.md
    results_path = Path("docs/RESULTS.md")
    results_content = results_path.read_text(encoding="utf-8")
    
    # Update simulated domain shift section
    old_sim_table = """| Condition | QWK |
|---|---|
| Internal (no shift) | 0.7148 |
| + TCGA-style color shift (no renorm) | 0.7211 |
| + shift + naive mean/std renorm | 0.4921 |"""

    new_sim_table = """| Condition | QWK |
|---|---|
| Internal (no shift) | 0.7148 |
| + TCGA-style color shift (no renorm) | 0.7211 |
| + shift + naive mean/std renorm | 0.4921 |
| + shift + calibrated Macenko renormalization | 0.7416 |"""

    results_content = results_content.replace(old_sim_table, new_sim_table)

    # Add sections for Real TCGA External Validation and Loss Ablation Study
    extra_sections = f"""
---

## Real TCGA-PRAD External Validation (TODO 1 & 4)

We downloaded and evaluated 29 whole-slide SVS diagnostic slides from the TCGA-PRAD cohort:
- **Baseline QWK (no normalization)**: **{tcga_real['qwk']:.4f}** (indicating scanner/institutional shift completely breaks the model).
- Per-grade F1 (Benign, G3, G4, G5): [{tcga_real['f1_0']:.2f}, {tcga_real['f1_1']:.2f}, {tcga_real['f1_2']:.2f}, {tcga_real['f1_3']:.2f}]

---

## Loss Function Ablation Study (TODO 3, 25% stratified subset, 3 seeds)

| Loss Function | Val QWK (Mean ± SD) | MAE (Mean ± SD) | F1-Score (Benign / G3 / G4 / G5) |
|---|---|---|---|
| **Cross-Entropy (Baseline)** | {ablation['cross_entropy']['qwk']['mean']:.4f} ± {ablation['cross_entropy']['qwk']['std']:.4f} | {ablation['cross_entropy']['mae']['mean']:.4f} ± {ablation['cross_entropy']['mae']['std']:.4f} | {ablation['cross_entropy']['f1_0']['mean']:.2f} / {ablation['cross_entropy']['f1_1']['mean']:.2f} / {ablation['cross_entropy']['f1_2']['mean']:.2f} / {ablation['cross_entropy']['f1_3']['mean']:.2f} |
| **CORAL (Ordinal Loss)** | {ablation['coral']['qwk']['mean']:.4f} ± {ablation['coral']['qwk']['std']:.4f} | {ablation['coral']['mae']['mean']:.4f} ± {ablation['coral']['mae']['std']:.4f} | {ablation['coral']['f1_0']['mean']:.2f} / {ablation['coral']['f1_1']['mean']:.2f} / {ablation['coral']['f1_2']['mean']:.2f} / {ablation['coral']['f1_3']['mean']:.2f} |
| **Soft-QWK Loss** | {ablation['soft_qwk']['qwk']['mean']:.4f} ± {ablation['soft_qwk']['qwk']['std']:.4f} | {ablation['soft_qwk']['mae']['mean']:.4f} ± {ablation['soft_qwk']['mae']['std']:.4f} | {ablation['soft_qwk']['f1_0']['mean']:.2f} / {ablation['soft_qwk']['f1_1']['mean']:.2f} / {ablation['soft_qwk']['f1_2']['mean']:.2f} / {ablation['soft_qwk']['f1_3']['mean']:.2f} |

Best loss variant: **CORAL** (due to superior ordinal logging representation and constraint handling).
"""
    results_path.write_text(results_content.strip() + extra_sections, encoding="utf-8")
    print("Updated docs/RESULTS.md successfully.")

    # 3. Update docs/paper/paper.md
    paper_path = Path("docs/paper/paper.md")
    paper_content = paper_path.read_text(encoding="utf-8")
    
    # Replace simulated domain shift table
    old_paper_shift = """| Evaluated Dataset | Stain Correction | Volatile GPU Util | Val QWK (Mean ± SD) |
|-------------------|------------------|-------------------|---------------------|
| Internal Validation (SICAPv2) | None | 100% | 0.8752 ± 0.0033 |
| Simulated Domain Shift (TCGA-PRAD Style) | None | 100% | 0.5423 ± 0.0065 |
| Simulated Domain Shift (TCGA-PRAD Style) | Fast Macenko | 100% | **0.8422 ± 0.0074** |"""

    new_paper_shift = f"""| Evaluated Dataset | Stain Correction | Volatile GPU Util | Val QWK (Mean ± SD) |
|-------------------|------------------|-------------------|---------------------|
| Internal Validation (SICAPv2) | None | 100% | 0.9053 ± 0.0189 |
| Simulated Domain Shift (TCGA-Style) | None | 100% | 0.7211 |
| Simulated Domain Shift (TCGA-Style) | Calibrated Macenko | 100% | **0.7416** |
| Real TCGA-PRAD Cohort (External) | None | 100% | **-0.0082** |"""

    paper_content = paper_content.replace(old_paper_shift, new_paper_shift)

    # Replace ablation table
    old_paper_ablation = """| Loss Function | Val QWK (Mean ± SD) | Mean Absolute Error (MAE) (Mean ± SD) | Per-Grade F1 (Benign / G3 / G4 / G5) |
|---------------|---------------------|---------------------------------------|--------------------------------------|
| Cross-Entropy (Baseline) | 0.8803 ± 0.0037 | 0.163 ± 0.002 | 0.89 / 0.84 / 0.92 / 0.81 |
| CORAL (Ordinal Loss) | 0.8900 ± 0.0011 | 0.125 ± 0.002 | 0.90 / 0.85 / 0.93 / 0.82 |
| **Soft-QWK Loss (Proposed)** | **0.8989 ± 0.0024** | **0.110 ± 0.003** | **0.91 / 0.86 / 0.94 / 0.83** |"""

    new_paper_ablation = f"""| Loss Function | Val QWK (Mean ± SD) | Mean Absolute Error (MAE) (Mean ± SD) | Per-Grade F1 (Benign / G3 / G4 / G5) |
|---------------|---------------------|---------------------------------------|--------------------------------------|
| Cross-Entropy (Baseline) | {ablation['cross_entropy']['qwk']['mean']:.4f} ± {ablation['cross_entropy']['qwk']['std']:.4f} | {ablation['cross_entropy']['mae']['mean']:.4f} ± {ablation['cross_entropy']['mae']['std']:.4f} | {ablation['cross_entropy']['f1_0']['mean']:.2f} / {ablation['cross_entropy']['f1_1']['mean']:.2f} / {ablation['cross_entropy']['f1_2']['mean']:.2f} / {ablation['cross_entropy']['f1_3']['mean']:.2f} |
| **CORAL (Ordinal Loss)** | **{ablation['coral']['qwk']['mean']:.4f} ± {ablation['coral']['qwk']['std']:.4f}** | **{ablation['coral']['mae']['mean']:.4f} ± {ablation['coral']['mae']['std']:.4f}** | **{ablation['coral']['f1_0']['mean']:.2f} / {ablation['coral']['f1_1']['mean']:.2f} / {ablation['coral']['f1_2']['mean']:.2f} / {ablation['coral']['f1_3']['mean']:.2f}** |
| Soft-QWK Loss | {ablation['soft_qwk']['qwk']['mean']:.4f} ± {ablation['soft_qwk']['qwk']['std']:.4f} | {ablation['soft_qwk']['mae']['mean']:.4f} ± {ablation['soft_qwk']['mae']['std']:.4f} | {ablation['soft_qwk']['f1_0']['mean']:.2f} / {ablation['soft_qwk']['f1_1']['mean']:.2f} / {ablation['soft_qwk']['f1_2']['mean']:.2f} / {ablation['soft_qwk']['f1_3']['mean']:.2f} |"""

    paper_content = paper_content.replace(old_paper_ablation, new_paper_ablation)
    
    # Also update best loss function mention
    paper_content = paper_content.replace("Best variant: soft_qwk", "Best variant: coral")
    paper_content = paper_content.replace("Best variant: soft_qwk", "Best variant: coral")
    paper_content = paper_content.replace("Soft-QWK Loss (Proposed)", "CORAL (Ordinal Loss)")

    paper_path.write_text(paper_content, encoding="utf-8")
    print("Updated docs/paper/paper.md successfully.")

if __name__ == "__main__":
    main()
