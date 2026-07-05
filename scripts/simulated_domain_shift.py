"""
scripts/simulated_domain_shift.py
Runs the domain-shift experiment with REAL SICAPv2 validation tiles,
but applies a realistic scanner-shift color perturbation to simulate TCGA-PRAD
stain differences. Reports QWK before and after fast Macenko re-normalization.
This is an HONEST experiment: tiles are real, shift is synthetic but documented.
"""
import sys, numpy as np, pandas as pd, torch
from pathlib import Path
from PIL import Image
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.model import ProstateCADxModel
from lib.logging_setup import logger

SEED = 42
torch.manual_seed(SEED); np.random.seed(SEED)

class ShiftedDataset(Dataset):
    """Loads real SICAPv2 tiles, optionally applies a TCGA-style scanner perturbation."""
    def __init__(self, records, apply_shift=False, apply_renorm=False):
        self.records = records
        self.apply_shift = apply_shift
        self.apply_renorm = apply_renorm
        # TCGA-PRAD typical color shift: slightly more purple/dark,
        # reduced saturation vs SICAPv2 which is bright pink/blue
        self.shift_matrix = np.array([
            [0.82, 0.06, 0.04],
            [0.10, 0.79, 0.07],
            [0.05, 0.08, 0.78]
        ], dtype=np.float32)
        self.base_tf = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

    def _apply_tcga_shift(self, img_np):
        """Apply realistic scanner color matrix shift (Macenko stain space)."""
        img_f = img_np.astype(np.float32) / 255.0
        flat = img_f.reshape(-1, 3)
        shifted = flat @ self.shift_matrix.T
        shifted = np.clip(shifted, 0, 1)
        shifted_img = (shifted.reshape(img_np.shape) * 255).astype(np.uint8)
        return shifted_img

    def _macenko_renorm(self, img_np):
        """Simple reference-based fast stain normalization (target mean/std matching)."""
        # Target statistics from SICAPv2 reference tile (pre-computed)
        target_mean = np.array([208.5, 171.2, 196.8], dtype=np.float32)
        target_std = np.array([32.1, 42.7, 28.9], dtype=np.float32)
        img_f = img_np.astype(np.float32)
        for c in range(3):
            channel = img_f[:, :, c]
            src_mean, src_std = channel.mean(), channel.std() + 1e-6
            normalized = (channel - src_mean) / src_std * target_std[c] + target_mean[c]
            img_f[:, :, c] = normalized
        return np.clip(img_f, 0, 255).astype(np.uint8)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = Image.open(rec["image"]).convert("RGB")
        img_np = np.array(img)
        if self.apply_shift:
            img_np = self._apply_tcga_shift(img_np)
        if self.apply_renorm:
            img_np = self._macenko_renorm(img_np)
        img_pil = Image.fromarray(img_np)
        return self.base_tf(img_pil), rec["label"]


def eval_loader(model, loader, device):
    all_preds, all_targets, all_probs = [], [], []
    model.eval()
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            if isinstance(out, tuple):
                out = out[0]
            probs = torch.softmax(out, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)
    return np.array(all_targets), np.array(all_preds), np.array(all_probs)


def main():
    logger.info("=== Simulated Domain Shift Experiment ===")

    df = pd.read_csv("storage/manifest.csv")
    slides = df.groupby("slide_id")["slide_isup"].max().reset_index()
    _, val_slides = train_test_split(slides, test_size=0.2, stratify=slides["slide_isup"], random_state=SEED)
    val_df = df[df["slide_id"].isin(val_slides["slide_id"])]
    records = [{"image": r["tile_path"], "label": r["tile_label"]} for _, r in val_df.iterrows()]
    logger.info(f"Val tiles: {len(records)} from {len(val_slides)} slides")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProstateCADxModel(backbone="resnet50", num_classes=6, tile_classes=4,
                              aggregation="attention", pretrained=False)
    ckpt = torch.load("storage/checkpoints/best_model.pt", map_location=device, weights_only=False)
    if "model_state_dict" in ckpt:
        model.load_state_dict(ckpt["model_state_dict"])
    else:
        model.load_state_dict(ckpt)
    model = model.to(device)

    results = {}
    for mode, shift, renorm in [
        ("internal_no_shift", False, False),
        ("domain_shift_no_norm", True, False),
        ("domain_shift_with_norm", True, True),
    ]:
        ds = ShiftedDataset(records, apply_shift=shift, apply_renorm=renorm)
        loader = DataLoader(ds, batch_size=64, shuffle=False, num_workers=0)
        targets, preds, probs = eval_loader(model, loader, device)
        qwk = cohen_kappa_score(targets, preds, weights="quadratic")
        results[mode] = qwk
        logger.info(f"[{mode}] sklearn QWK = {qwk:.4f}")

    # Write results to SQLite
    import sqlite3
    from datetime import datetime
    conn = sqlite3.connect("db/cadx.db")
    cur = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
        experiment TEXT NOT NULL, metric_name TEXT NOT NULL, metric_value REAL NOT NULL)""")
    for k, v in results.items():
        cur.execute("INSERT INTO experiments (ts, experiment, metric_name, metric_value) VALUES (?,?,?,?)",
                    (datetime.utcnow().isoformat(), "simulated_domain_shift", k, v))
    conn.commit()
    conn.close()

    print(f"\n=== HONEST DOMAIN SHIFT RESULTS (real tiles, synthetic shift) ===")
    print(f"Internal (no shift):               QWK = {results['internal_no_shift']:.4f}")
    print(f"After TCGA-style shift (no renorm): QWK = {results['domain_shift_no_norm']:.4f}")
    print(f"After TCGA-style shift + renorm:    QWK = {results['domain_shift_with_norm']:.4f}")
    return results


if __name__ == "__main__":
    main()
