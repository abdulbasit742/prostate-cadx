"""
scripts/temperature_scaling.py
TODO 2: Fit temperature scaling on the validation set, compute ECE before/after,
regenerate the reliability diagram, and recompute risk-coverage with calibrated confidence.
"""
import sys, numpy as np, pandas as pd, torch, torch.nn as nn
from pathlib import Path
from datetime import datetime
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.model import ProstateCADxModel
from lib.data import GleasonDataset, get_augmentations
from lib.logging_setup import logger

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS  = [42, 43, 44]
NUM_CLASSES = 4  # tile labels 0-3


# --------------------------------------------------------------------------- #
# ECE helper
# --------------------------------------------------------------------------- #
def compute_ece(probs, targets, n_bins=10):
    """Expected Calibration Error."""
    confidences = probs.max(axis=1)
    preds       = probs.argmax(axis=1)
    correct     = (preds == targets).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece  = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        acc  = correct[mask].mean()
        conf = confidences[mask].mean()
        ece += mask.sum() / len(probs) * abs(acc - conf)
    return float(ece)


# --------------------------------------------------------------------------- #
# Reliability diagram
# --------------------------------------------------------------------------- #
def plot_reliability(probs_raw, probs_cal, targets, path, n_bins=10):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor("#0d1117")
    for ax, probs, title in [(axes[0], probs_raw, "Before Calibration"),
                             (axes[1], probs_cal, "After Temperature Scaling")]:
        ax.set_facecolor("#0d1117")
        confs = probs.max(axis=1)
        preds = probs.argmax(axis=1)
        correct = (preds == targets).astype(float)
        bins = np.linspace(0, 1, n_bins + 1)
        bin_accs, bin_confs, bin_ns = [], [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (confs >= lo) & (confs < hi)
            if mask.sum() == 0:
                bin_accs.append(0); bin_confs.append((lo+hi)/2); bin_ns.append(0)
                continue
            bin_accs.append(correct[mask].mean())
            bin_confs.append(confs[mask].mean())
            bin_ns.append(mask.sum())
        ece = compute_ece(probs, targets)
        ax.plot([0,1],[0,1],"--",color="#666",linewidth=1.5, label="Perfect calibration")
        ax.bar([(b[0]+b[1])/2 for b in zip(bins[:-1],bins[1:])],
               bin_accs, width=0.1, alpha=0.7, color="#2196F3", label="Accuracy per bin")
        ax.plot(bin_confs, bin_accs, "o-", color="#FF9800", linewidth=2, markersize=6, label=f"ECE={ece:.4f}")
        ax.set_xlabel("Mean Confidence", color="white"); ax.set_ylabel("Accuracy", color="white")
        ax.set_title(title, color="white", fontweight="bold")
        ax.tick_params(colors="white"); ax.legend(facecolor="#1a1a2e", labelcolor="white")
        ax.spines[:].set_color("#333")
    plt.tight_layout()
    plt.savefig(path, dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    logger.info(f"Reliability diagram saved: {path}")


# --------------------------------------------------------------------------- #
# Load model + get raw logits
# --------------------------------------------------------------------------- #
def load_model():
    model = ProstateCADxModel(backbone="resnet50", num_classes=6, tile_classes=4,
                              aggregation="attention", pretrained=False)
    ckpt = torch.load("storage/checkpoints/best_model.pt", map_location=DEVICE, weights_only=False)
    state = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    return model.to(DEVICE).eval()


def get_logits_and_targets(model, records):
    _, val_tf = get_augmentations(256)
    dataset = GleasonDataset(records, transform=val_tf)
    loader  = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=0)
    logits_list, targets_list = [], []
    with torch.no_grad():
        for imgs, tgts in loader:
            out = model(imgs.to(DEVICE))
            if isinstance(out, tuple): out = out[0]
            logits_list.append(out.cpu())
            targets_list.append(tgts)
    return torch.cat(logits_list), torch.cat(targets_list)


# --------------------------------------------------------------------------- #
# Temperature scaling (optimize NLL)
# --------------------------------------------------------------------------- #
def fit_temperature(logits, targets):
    T = nn.Parameter(torch.ones(1))
    optimizer = torch.optim.LBFGS([T], lr=0.01, max_iter=500)
    ce = nn.CrossEntropyLoss()
    def closure():
        optimizer.zero_grad()
        loss = ce(logits / T, targets)
        loss.backward()
        return loss
    optimizer.step(closure)
    return T.item()


# --------------------------------------------------------------------------- #
# Risk-coverage with calibrated confidence
# --------------------------------------------------------------------------- #
def risk_coverage(targets_np, preds_np, confs_np):
    sorted_idx = np.argsort(-confs_np)
    results = {}
    for cov in [1.0, 0.9, 0.8]:
        n = max(2, int(len(sorted_idx) * cov))
        sel = sorted_idx[:n]
        if len(np.unique(targets_np[sel])) < 2:
            results[cov] = float("nan")
        else:
            results[cov] = cohen_kappa_score(targets_np[sel], preds_np[sel], weights="quadratic")
    return results


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    logger.info("=== Temperature Scaling + ECE Calibration (3 seeds) ===")
    model = load_model()
    df    = pd.read_csv("storage/manifest.csv")

    seed_results = {
        "T": [], "ece_before": [], "ece_after": [],
        "qwk_100_cal": [], "qwk_90_cal": [], "qwk_80_cal": [],
    }

    for seed in SEEDS:
        torch.manual_seed(seed); np.random.seed(seed)
        slides = df.groupby("slide_id")["slide_isup"].max().reset_index()
        _, val_slides = train_test_split(slides, test_size=0.2,
                                         stratify=slides["slide_isup"], random_state=seed)
        val_df  = df[df["slide_id"].isin(val_slides["slide_id"])]
        records = [{"image": r["tile_path"], "label": r["tile_label"]} for _, r in val_df.iterrows()]
        logger.info(f"Seed {seed}: {len(records)} val tiles")

        logits, targets = get_logits_and_targets(model, records)
        targets_np = targets.numpy()

        # Raw (uncalibrated)
        probs_raw = torch.softmax(logits, dim=1).numpy()
        ece_before = compute_ece(probs_raw, targets_np)
        logger.info(f"Seed {seed}: ECE before = {ece_before:.4f}")

        # Fit temperature
        T = fit_temperature(logits.clone().requires_grad_(True), targets)
        logger.info(f"Seed {seed}: Optimal T = {T:.4f}")

        # Calibrated
        probs_cal = torch.softmax(logits / T, dim=1).numpy()
        ece_after = compute_ece(probs_cal, targets_np)
        logger.info(f"Seed {seed}: ECE after  = {ece_after:.4f}")

        preds_cal = probs_cal.argmax(axis=1)
        confs_cal = probs_cal.max(axis=1)
        rc = risk_coverage(targets_np, preds_cal, confs_cal)
        qwk_100 = cohen_kappa_score(targets_np, preds_cal, weights="quadratic")
        logger.info(f"Seed {seed}: Calibrated QWK@100={qwk_100:.4f} @90={rc[0.9]:.4f} @80={rc[0.8]:.4f}")

        seed_results["T"].append(T)
        seed_results["ece_before"].append(ece_before)
        seed_results["ece_after"].append(ece_after)
        seed_results["qwk_100_cal"].append(qwk_100)
        seed_results["qwk_90_cal"].append(rc[0.9])
        seed_results["qwk_80_cal"].append(rc[0.8])

        # Reliability diagram for seed 42
        if seed == 42:
            plot_reliability(probs_raw, probs_cal, targets_np,
                             "docs/assets/reliability_diagram.png")

    # Stats
    stats = {k: (np.nanmean(v), np.nanstd(v)) for k, v in seed_results.items()}

    print("\n=== CALIBRATION RESULTS (3 seeds) ===")
    print(f"Optimal temperature T:    {stats['T'][0]:.4f} ± {stats['T'][1]:.4f}")
    print(f"ECE before calibration:   {stats['ece_before'][0]:.4f} ± {stats['ece_before'][1]:.4f}")
    print(f"ECE after calibration:    {stats['ece_after'][0]:.4f} ± {stats['ece_after'][1]:.4f}")
    print(f"Calibrated QWK @100%:     {stats['qwk_100_cal'][0]:.4f} ± {stats['qwk_100_cal'][1]:.4f}")
    print(f"Calibrated QWK @90%:      {stats['qwk_90_cal'][0]:.4f} ± {stats['qwk_90_cal'][1]:.4f}")
    print(f"Calibrated QWK @80%:      {stats['qwk_80_cal'][0]:.4f} ± {stats['qwk_80_cal'][1]:.4f}")

    # Log to SQLite
    import sqlite3
    conn = sqlite3.connect("db/cadx.db")
    cur  = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
        experiment TEXT NOT NULL, metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL)""")
    for k, (m, s) in stats.items():
        cur.execute("INSERT INTO experiments (ts,experiment,metric_name,metric_value) VALUES (?,?,?,?)",
                    (datetime.utcnow().isoformat(), "calibration", f"{k}_mean", m))
        cur.execute("INSERT INTO experiments (ts,experiment,metric_name,metric_value) VALUES (?,?,?,?)",
                    (datetime.utcnow().isoformat(), "calibration", f"{k}_std",  s))
    conn.commit(); conn.close()

    # Save temperature for later use
    import json
    Path("storage/calibration.json").write_text(
        json.dumps({"temperature": float(stats["T"][0]),
                    "ece_before": float(stats["ece_before"][0]),
                    "ece_after": float(stats["ece_after"][0])}, indent=2)
    )
    logger.info("Saved storage/calibration.json")
    return stats

if __name__ == "__main__":
    main()
