"""
scripts/loss_ablation_train.py
TODO 3: Actually train three model variants on the SAME corrected leak-free split,
same seeds, same epochs, same hyperparameters — varying ONLY the loss function.

Variants:
  (a) cross_entropy  — standard CE
  (b) coral          — CORN/CORAL ordinal regression
  (c) soft_qwk       — differentiable soft-QWK loss

Reports QWK, MAE, per-grade F1 for each variant as mean ± std over 3 seeds.
Logs to SQLite experiment="loss_ablation_real".
"""
import sys, os, time, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torch.optim as optim
from pathlib import Path
from datetime import datetime
from sklearn.metrics import cohen_kappa_score, f1_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.model import ProstateCADxModel
from lib.data import GleasonDataset, get_augmentations
from lib.logging_setup import logger

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
SEEDS     = [42, 43, 44]
EPOCHS    = 5          # short ablation runs (not full training) — honest caveat noted
LR        = 2e-4
WD        = 0.05
BATCH     = 64
NUM_C     = 4          # tile label classes (0-3)


# ============================================================
# Loss Functions
# ============================================================
class CORNLoss(nn.Module):
    """CORN (Conditional Ordinal Regression for Neural Nets) loss."""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, logits, labels):
        # Take first num_classes logits
        logits = logits[:, :self.num_classes]
        sets   = []
        n      = self.num_classes - 1
        for i in range(n):
            label_i = (labels > i).long()
            sets.append(torch.clamp(logits[:, i], -20, 20))
        if not sets:
            return nn.CrossEntropyLoss()(logits, labels)
        loss = 0.0
        for i, lgt in enumerate(sets):
            label_i = (labels > i).float()
            loss    = loss + nn.BCEWithLogitsLoss()(lgt, label_i)
        return loss / max(1, len(sets))


class SoftQWKLoss(nn.Module):
    """Differentiable soft-QWK loss (optimizes quadratic weighted kappa)."""
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # Weight matrix W[i,j] = (i-j)^2 / (num_classes-1)^2
        W = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                W[i, j] = (i - j)**2 / (num_classes - 1)**2
        self.register_buffer("W", W)

    def forward(self, logits, labels):
        logits = logits[:, :self.num_classes]
        probs  = torch.softmax(logits, dim=1)             # (B, C)
        oh     = nn.functional.one_hot(labels, self.num_classes).float()  # (B, C)
        # Numerator: E[w * P(pred) * P(true)]
        O = torch.matmul(oh.T, probs)                     # (C, C)
        hist_p = probs.mean(0, keepdim=True)               # (1, C)
        hist_t = oh.mean(0, keepdim=True)                  # (1, C)
        E = torch.matmul(hist_t.T, hist_p)                 # (C, C)
        num = (self.W * O).sum()
        den = (self.W * E).sum() + 1e-8
        return num / den


# ============================================================
# Helpers
# ============================================================
def get_split(seed):
    df = pd.read_csv("storage/manifest.csv")
    slides = df.groupby("slide_id")["slide_isup"].max().reset_index()
    train_slides, val_slides = train_test_split(
        slides, test_size=0.2, stratify=slides["slide_isup"], random_state=seed)
    # Take a 25% stratified subset of the train slides to speed up the ablation study
    train_slides_sub, _ = train_test_split(
        train_slides, test_size=0.75, stratify=train_slides["slide_isup"], random_state=seed)
    train_df = df[df["slide_id"].isin(train_slides_sub["slide_id"])]
    val_df   = df[df["slide_id"].isin(val_slides["slide_id"])]
    return train_df, val_df


def make_loader(df, train, batch_size=BATCH):
    records = [{"image": r["tile_path"], "label": r["tile_label"]} for _, r in df.iterrows()]
    train_tf, val_tf = get_augmentations(256)
    transform = train_tf if train else val_tf
    ds = GleasonDataset(records, transform=transform)
    return DataLoader(ds, batch_size=batch_size, shuffle=train, num_workers=0,
                      drop_last=train, pin_memory=True)


def train_one_epoch(model, loader, optimizer, criterion, scaler=None):
    model.train()
    total_loss = 0.0
    for imgs, targets in loader:
        imgs, targets = imgs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        if scaler:
            with torch.cuda.amp.autocast():
                out = model(imgs)
                if isinstance(out, tuple): out = out[0]
                out = out[:, :NUM_C]
                loss = criterion(out, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            out = model(imgs)
            if isinstance(out, tuple): out = out[0]
            out = out[:, :NUM_C]
            loss = criterion(out, targets)
            loss.backward()
            optimizer.step()
        total_loss += loss.item()
    return total_loss / max(1, len(loader))


def evaluate(model, loader, loss_name="cross_entropy"):
    model.eval()
    all_preds, all_targets, all_probs = [], [], []
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(DEVICE)
            out  = model(imgs)
            if isinstance(out, tuple): out = out[0]
            out = out[:, :NUM_C]
            if loss_name == "coral":
                logits = out[:, :NUM_C - 1]
                preds = (logits > 0).sum(dim=1).cpu().numpy()
                probs = torch.sigmoid(logits).cpu().numpy()
            else:
                probs = torch.softmax(out, dim=1).cpu().numpy()
                preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


# ============================================================
# Train single variant, single seed
# ============================================================
def run_variant(loss_name, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    train_df, val_df = get_split(seed)
    logger.info(f"[{loss_name}|seed={seed}] train={len(train_df)} tiles val={len(val_df)} tiles")

    train_loader = make_loader(train_df, train=True)
    val_loader   = make_loader(val_df,   train=False)

    model = ProstateCADxModel(backbone="resnet50", num_classes=6, tile_classes=4,
                               aggregation="attention", pretrained=True)
    model = model.to(DEVICE)
    # Use channels_last for GPU efficiency
    model = model.to(memory_format=torch.channels_last)

    if loss_name == "cross_entropy":
        criterion = nn.CrossEntropyLoss()
    elif loss_name == "coral":
        criterion = CORNLoss(NUM_C)
    elif loss_name == "soft_qwk":
        criterion = SoftQWKLoss(NUM_C).to(DEVICE)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler    = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None

    best_qwk = -1.0
    best_ckpt = f"storage/checkpoints/ablation_{loss_name}_seed{seed}.pt"

    for epoch in range(1, EPOCHS + 1):
        t0   = time.time()
        loss = train_one_epoch(model, train_loader, optimizer, criterion, scaler)
        scheduler.step()
        targets, preds, probs = evaluate(model, val_loader, loss_name)
        if len(np.unique(targets)) >= 2:
            qwk = cohen_kappa_score(targets, preds, weights="quadratic")
        else:
            qwk = 0.0
        dt = time.time() - t0
        logger.info(f"  [{loss_name}|seed={seed}|ep{epoch}/{EPOCHS}] loss={loss:.4f} val_qwk={qwk:.4f} ({dt:.0f}s)")
        if qwk > best_qwk:
            best_qwk = qwk
            torch.save({"model_state_dict": model.state_dict(), "qwk": qwk}, best_ckpt)

    # Final eval on best checkpoint
    ckpt = torch.load(best_ckpt, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    targets, preds, _ = evaluate(model, val_loader, loss_name)
    qwk = cohen_kappa_score(targets, preds, weights="quadratic") \
          if len(np.unique(targets)) >= 2 else float("nan")
    mae = mean_absolute_error(targets, preds)
    f1s = f1_score(targets, preds, average=None, labels=[0,1,2,3], zero_division=0)

    logger.info(f"  [{loss_name}|seed={seed}] FINAL QWK={qwk:.4f} MAE={mae:.4f}")
    return {"qwk": qwk, "mae": mae, "f1_0": f1s[0], "f1_1": f1s[1],
            "f1_2": f1s[2], "f1_3": f1s[3]}


# ============================================================
# Main
# ============================================================
def main():
    logger.info("=== TODO 3: Real Loss Ablation Training ===")
    logger.info(f"Epochs per variant: {EPOCHS} (short ablation — noted as caveat)")
    logger.info(f"Variants: cross_entropy, coral, soft_qwk | Seeds: {SEEDS}")

    all_results = {}
    for loss_name in ["cross_entropy", "coral", "soft_qwk"]:
        seed_results = {k: [] for k in ["qwk","mae","f1_0","f1_1","f1_2","f1_3"]}
        for seed in SEEDS:
            r = run_variant(loss_name, seed)
            for k in seed_results:
                seed_results[k].append(r[k])
        stats = {k: (np.nanmean(v), np.nanstd(v)) for k, v in seed_results.items()}
        all_results[loss_name] = stats
        logger.info(f"[{loss_name}] QWK={stats['qwk'][0]:.4f}±{stats['qwk'][1]:.4f} "
                    f"MAE={stats['mae'][0]:.4f}±{stats['mae'][1]:.4f}")

    print("\n=== REAL LOSS ABLATION RESULTS (3 seeds each) ===")
    print(f"{'Loss':>15} | {'QWK Mean':>10} | {'QWK Std':>9} | {'MAE Mean':>9} | {'F1 ISUP1':>9}")
    print("-" * 65)
    for loss_name, stats in all_results.items():
        print(f"{loss_name:>15} | {stats['qwk'][0]:>10.4f} | {stats['qwk'][1]:>9.4f} "
              f"| {stats['mae'][0]:>9.4f} | {stats['f1_1'][0]:>9.4f}")

    # Best variant
    best = max(all_results, key=lambda k: all_results[k]["qwk"][0])
    print(f"\nBest variant: {best} (QWK={all_results[best]['qwk'][0]:.4f})")

    # Log to SQLite
    import sqlite3
    conn = sqlite3.connect("db/cadx.db")
    cur  = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
        experiment TEXT NOT NULL, metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL)""")
    now = datetime.utcnow().isoformat()
    for loss_name, stats in all_results.items():
        for k, (m, s) in stats.items():
            cur.execute("INSERT INTO experiments VALUES (NULL,?,?,?,?)",
                        (now, "loss_ablation_real", f"{loss_name}_{k}_mean", m))
            cur.execute("INSERT INTO experiments VALUES (NULL,?,?,?,?)",
                        (now, "loss_ablation_real", f"{loss_name}_{k}_std", s))
    conn.commit(); conn.close()

    # Save JSON for paper autofill
    out = {ln: {k: {"mean": float(v[0]), "std": float(v[1])} for k, v in st.items()}
           for ln, st in all_results.items()}
    Path("storage/loss_ablation_results.json").write_text(json.dumps(out, indent=2))
    logger.info("Saved storage/loss_ablation_results.json")
    return all_results

if __name__ == "__main__":
    main()
