"""
Skill 101: Results Writer
Reads ALL metrics from SQLite and writes a full experiments table to docs/RESULTS.md.
Triggered after every training cycle.
"""
import sys
import sqlite3
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

DOCS = Path("docs")
ASSETS = DOCS / "assets"
DOCS.mkdir(exist_ok=True)
ASSETS.mkdir(exist_ok=True)

def fetch_metrics(db_path="db/cadx.db"):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("""
        SELECT id, ts, kappa, val_loss, batch_size, epoch, checkpoint_path
        FROM metrics
        ORDER BY ts ASC
    """)
    rows = cur.fetchall()
    conn.close()
    return rows

def classify_run(ts_str, checkpoint_path):
    """Determine if the row belongs to synthetic or real-data training."""
    # Real data runs started after 2026-07-04 11:00 UTC
    if ts_str >= "2026-07-04T11:":
        return "Real (SICAPv2)"
    return "Synthetic (Fallback)"

def format_kappa(k):
    if k is None:
        return "N/A"
    return f"{k:.4f}"

def write_results(rows):
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    # Find best real-data run
    real_rows = [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows if classify_run(r[1], r[6]) == "Real (SICAPv2)"]
    best_row = max(real_rows, key=lambda r: r[2] if r[2] is not None else -999, default=None)

    lines = [
        "# Prostate CADx — Experiment Results",
        "",
        f"> Auto-generated: {now}",
        "",
        "## Summary (Best Real-Data Run)",
        "",
    ]

    if best_row:
        lines += [
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| **Best Val QWK (Kappa)** | **{format_kappa(best_row[2])}** |",
            f"| Best Val Loss | {format_kappa(best_row[3])} |",
            f"| Epoch | {best_row[5]} |",
            f"| Batch Size | {best_row[4]} |",
            f"| Run Type | Real (SICAPv2) |",
            f"| Dataset | CrowdGleason/SICAPv2 (Zenodo 14178894) |",
            f"| N Tiles Train | 10,528 |",
            f"| N Tiles Val | 3,719 |",
            "",
        ]

    lines += [
        "---",
        "",
        "## All Experiments",
        "",
        "| ID | Timestamp (UTC) | Run Type | Epoch | Batch | Val Kappa | Val Loss |",
        "|----|-----------------|----------|-------|-------|-----------|----------|",
    ]

    for row in rows:
        rid, ts, kappa, vloss, bs, epoch, ckpt = row
        run_type = classify_run(ts, ckpt)
        ts_short = ts[:19].replace("T", " ")
        lines.append(
            f"| {rid} | {ts_short} | {run_type} | {epoch} | {bs} "
            f"| {format_kappa(kappa)} | {format_kappa(vloss)} |"
        )

    lines += [
        "",
        "---",
        "",
        "## Ablation: Synthetic vs Real Data",
        "",
        "| Metric | Synthetic (Fallback) | Real (SICAPv2) |",
        "|--------|---------------------|----------------|",
    ]

    if real_rows:
        real_kappas = [r[2] for r in real_rows if r[2] is not None]
        syn_rows = [(r[0], r[1], r[2], r[3], r[4], r[5]) for r in rows if classify_run(r[1], r[6]) == "Synthetic (Fallback)"]
        syn_kappas = [r[2] for r in syn_rows if r[2] is not None]

        best_real = max(real_kappas) if real_kappas else 0
        best_syn = max(syn_kappas) if syn_kappas else 0
        mean_real = sum(real_kappas) / len(real_kappas) if real_kappas else 0
        mean_syn = sum(syn_kappas) / len(syn_kappas) if syn_kappas else 0

        lines += [
            f"| Best Val Kappa | {format_kappa(best_syn)} | **{format_kappa(best_real)}** |",
            f"| Mean Val Kappa | {format_kappa(mean_syn)} | {format_kappa(mean_real)} |",
            f"| N Experiments | {len(syn_rows)} | {len(real_rows)} |",
        ]

    lines += [
        "",
        "---",
        "",
        "## Risk-Coverage Analysis",
        "",
        "> Risk-coverage curve measures QWK on the subset of cases where the model confidence",
        "> exceeds a threshold (80% / 90%). High-confidence predictions are expected to have",
        "> substantially better Kappa than the full test set.",
        "",
        "| Coverage | Expected QWK | Note |",
        "|----------|-------------|------|",
        "| 100% (all) | [FILL_KAPPA_100] | Full validation set |",
        "| 90% (top 90%) | [FILL_KAPPA_90] | Pending eval run |",
        "| 80% (top 80%) | [FILL_KAPPA_80] | Pending eval run |",
        "",
        "---",
        "",
        "## Visualizations",
        "",
        "### Training Kappa Curve",
        "![Kappa Curve](assets/kappa_curve.png)",
        "",
        "### Confusion Matrix (Best Model)",
        "![Confusion Matrix](assets/confusion_matrix.png)",
        "",
        "### Per-Grade F1 Scores",
        "![Per-Grade F1](assets/per_grade_f1.png)",
        "",
        "### Attention Map Overlap Score",
        "![Attention Heatmap](assets/attention_overlap.png)",
        "",
    ]

    out_path = DOCS / "RESULTS.md"
    out_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[results_writer] Written {len(rows)} experiment rows to {out_path}")
    return best_row

if __name__ == "__main__":
    rows = fetch_metrics()
    write_results(rows)
    print("Done.")
