"""
scripts/tcga_download_and_eval.py
TODO 1 — Download 30 open-access TCGA-PRAD diagnostic SVS slides using the local manifest and clinical JSON.
Tile them, evaluate them using the best SICAPv2 checkpoint, and log results.
"""
import sys
import json
import os
import time
import urllib.request
import urllib.parse
import hashlib
import sqlite3
import numpy as np
import pandas as pd
import torch
from pathlib import Path
from datetime import datetime
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as T

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.model import ProstateCADxModel
from lib.data import GleasonDataset, get_augmentations
from lib.logging_setup import logger

DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
TILE_SIZE  = 256
TISSUE_THR = 0.05          # min tissue fraction to keep tile
MAX_TILES  = 100           # max tiles per slide (for speed and storage)
TARGET_SLIDES = 30
TCGA_DIR   = Path("storage/data/tcga_prad")
CACHE_DIR  = Path("storage/data/tcga_tiles")
TCGA_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

MANIFEST_PATH = Path("C:/Users/absh5/Downloads/gdc_manifest.2026-07-04.093745.txt")
CLINICAL_PATH = Path("C:/Users/absh5/Downloads/clinical.project-tcga-prad.2026-07-04.json")

# ============================================================
# Mapping function
# ============================================================
def map_gleason_to_isup(p, s, score):
    try: p_val = int(p.split()[-1]) if p else None
    except: p_val = None
    try: s_val = int(s.split()[-1]) if s else None
    except: s_val = None

    if p_val is None or s_val is None:
        if score == 6: return 1
        if score == 8: return 4
        if score in (9, 10): return 5
        return None
    if p_val <= 3 and s_val <= 3: return 1
    if p_val == 3 and s_val == 4: return 2
    if p_val == 4 and s_val == 3: return 3
    if p_val == 4 and s_val == 4: return 4
    if (p_val == 3 and s_val == 5) or (p_val == 5 and s_val == 3): return 4
    if (p_val == 4 and s_val == 5) or (p_val == 5 and s_val == 4) or (p_val == 5 and s_val == 5): return 5
    if p_val + s_val <= 6: return 1
    if p_val + s_val == 8: return 4
    if p_val + s_val >= 9: return 5
    return None

def isup_to_label(isup):
    # Map TCGA ISUP grade to SICAPv2 tile_label (0-3).
    # SICAPv2: 0=NC/benign, 1=G3, 2=G4, 3=G5
    mapping = {0: 0, 1: 1, 4: 2, 5: 3}
    return mapping.get(isup, None)

# ============================================================
# GDC Download helper
# ============================================================
def download_svs(file_id, dest_path, fname):
    url = f"https://api.gdc.cancer.gov/data/{file_id}"
    if dest_path.exists() and dest_path.stat().st_size > 1e6:
        logger.info(f"  Already downloaded: {fname}")
        return True
    logger.info(f"  Downloading {fname} ({dest_path}) ...")
    try:
        req = urllib.request.urlopen(url, timeout=120)
        with open(dest_path, "wb") as f:
            while True:
                chunk = req.read(1 << 20)  # 1 MB chunks
                if not chunk: break
                f.write(chunk)
        logger.info(f"  Done: {dest_path.stat().st_size/1e6:.1f} MB")
        return True
    except Exception as e:
        logger.warning(f"  Download failed: {e}")
        if dest_path.exists(): dest_path.unlink()
        return False

# ============================================================
# Slide Tiler
# ============================================================
def tile_svs(svs_path, out_dir, slide_id):
    try:
        import tiffslide
        from skimage.filters import threshold_otsu
        from skimage.color import rgb2gray
    except ImportError as e:
        logger.error(f"Missing: {e}")
        return []

    tiles_dir = out_dir / slide_id
    tiles_dir.mkdir(exist_ok=True)
    existing = list(tiles_dir.glob("*.jpg"))
    if len(existing) >= 10:
        logger.info(f"  Tiles cached: {len(existing)} for {slide_id}")
        return [str(p) for p in existing[:MAX_TILES]]

    try:
        slide = tiffslide.open_slide(str(svs_path))
    except Exception as e:
        logger.warning(f"  Cannot open {svs_path}: {e}")
        return []

    w, h = slide.dimensions
    logger.info(f"  Slide {slide_id}: {w}x{h} px")

    # Use a thumbnail for tissue mask
    thumb_scale = 32
    thumb_w, thumb_h = max(1, w // thumb_scale), max(1, h // thumb_scale)
    thumb = np.array(slide.get_thumbnail((thumb_w, thumb_h)).convert("RGB"))
    gray  = rgb2gray(thumb)
    try:
        thr = threshold_otsu(gray)
    except Exception:
        thr = 0.8
    tissue_mask = gray < thr  # dark = tissue

    tile_paths = []
    from PIL import Image
    count = 0
    step  = TILE_SIZE
    for y in range(0, h - TILE_SIZE, step):
        for x in range(0, w - TILE_SIZE, step):
            if count >= MAX_TILES:
                break
            tx = int(x / thumb_scale); ty = int(y / thumb_scale)
            tw = max(1, TILE_SIZE // thumb_scale); th_ = max(1, TILE_SIZE // thumb_scale)
            region = tissue_mask[ty:ty+th_, tx:tx+tw]
            if region.size == 0 or region.mean() < TISSUE_THR:
                continue
            try:
                tile = slide.read_region((x, y), 0, (TILE_SIZE, TILE_SIZE)).convert("RGB")
                fname = tiles_dir / f"{slide_id}_{x}_{y}.jpg"
                tile.save(str(fname), quality=90)
                tile_paths.append(str(fname))
                count += 1
            except Exception:
                continue
        if count >= MAX_TILES:
            break

    slide.close()
    logger.info(f"  Tiled {count} tiles from {slide_id}")
    return tile_paths

class TilePredDataset(Dataset):
    def __init__(self, paths, transform):
        self.paths = paths
        self.transform = transform

    def __len__(self): return len(self.paths)

    def __getitem__(self, idx):
        from PIL import Image
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img), 0

def eval_tiles(model, tile_paths):
    tf = T.Compose([
        T.Resize((256, 256)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    ds = TilePredDataset(tile_paths, tf)
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    probs_all = []
    model.eval()
    with torch.no_grad():
        for imgs, _ in loader:
            out = model(imgs.to(DEVICE))
            if isinstance(out, tuple): out = out[0]
            probs = torch.softmax(out, dim=1).cpu().numpy()
            probs_all.append(probs)
    if not probs_all:
        return None
    return np.concatenate(probs_all, axis=0)

def slide_level_pred(tile_probs):
    mean_probs = tile_probs.mean(axis=0)
    return int(np.argmax(mean_probs))

# ============================================================
# Main
# ============================================================
def main():
    logger.info("=== TODO 1: Real TCGA-PRAD External Validation ===")
    import tiffslide

    # 1. Parse clinical data
    if not CLINICAL_PATH.exists():
        logger.error(f"Clinical file missing at {CLINICAL_PATH}")
        return
    with open(CLINICAL_PATH, 'r', encoding='utf-8') as f:
        clin = json.load(f)

    clin_map = {}
    for r in clin:
        sub_id = r['submitter_id']
        for d in r.get('diagnoses', []):
            if d.get('tissue_or_organ_of_origin') == 'Prostate gland':
                clin_map[sub_id] = {
                    'isup': map_gleason_to_isup(d.get('primary_gleason_grade'), d.get('secondary_gleason_grade'), d.get('gleason_score'))
                }

    # 2. Parse manifest
    if not MANIFEST_PATH.exists():
        logger.error(f"Manifest file missing at {MANIFEST_PATH}")
        return
    df_man = pd.read_csv(MANIFEST_PATH, sep='\t')
    svs_df = df_man[df_man['filename'].str.endswith('.svs', na=False)].copy()

    # 3. Filter eligible diagnostic slides
    eligible_slides = []
    for idx, row in svs_df.iterrows():
        fn = row['filename']
        pat_id = '-'.join(fn.split('-')[:3])
        if pat_id in clin_map:
            isup = clin_map[pat_id]['isup']
            if isup in [1, 4, 5]:
                is_dx = any(x in fn for x in ['-DX1', '-DX2', '-DX3', '-DX4', '-DX5'])
                if is_dx:
                    eligible_slides.append({
                        'id': row['id'],
                        'filename': fn,
                        'size': row['size'],
                        'isup': isup
                    })

    df_elig = pd.DataFrame(eligible_slides)
    df_selected = df_elig.sort_values(by='size').head(TARGET_SLIDES)
    logger.info(f"Selected {len(df_selected)} smallest diagnostic slides. Total size: {df_selected['size'].sum()/1e6:.1f} MB")

    # 4. Load model
    model = ProstateCADxModel(backbone="resnet50", num_classes=6, tile_classes=4,
                               aggregation="attention", pretrained=False)
    ckpt = torch.load("storage/checkpoints/best_model.pt", map_location=DEVICE, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model = model.to(DEVICE).eval()
    logger.info("Model checkpoint loaded.")

    slide_results = []
    eval_ok        = 0

    for idx, row in df_selected.iterrows():
        fid = row['id']
        fname = row['filename']
        isup = row['isup']
        label = isup_to_label(isup)

        logger.info(f"[{eval_ok+1}/{TARGET_SLIDES}] {fname} ({row['size']/1e6:.1f} MB)")

        dest = TCGA_DIR / fname
        
        # Verify and download logic
        verification_passed = False
        for attempt in range(2):
            if dest.exists() and dest.stat().st_size > 1e6:
                try:
                    # Test open and force loading pages
                    test_slide = tiffslide.open_slide(str(dest))
                    _ = test_slide.dimensions
                    test_slide.close()
                    verification_passed = True
                    break
                except Exception as e:
                    logger.warning(f"  Existing file corrupted: {fname}. Re-downloading (error: {e})")
                    try:
                        dest.unlink()
                    except Exception:
                        pass
            
            # Download
            download_svs(fid, dest, fname)
        
        if not verification_passed:
            logger.error(f"  Failed to get a valid slide file for {fname} after download attempts.")
            continue

        slide_id = Path(fname).stem[:30]
        tile_paths = tile_svs(dest, CACHE_DIR, slide_id)
        if len(tile_paths) < 5:
            logger.warning(f"  Too few tiles ({len(tile_paths)}) — skipping evaluation")
            continue

        tile_probs = eval_tiles(model, tile_paths)
        if tile_probs is None:
            continue

        slide_pred = slide_level_pred(tile_probs)
        eval_ok += 1
        slide_results.append({
            "fname": fname,
            "isup_true": isup,
            "label_true": label,
            "slide_pred": slide_pred,
            "n_tiles": len(tile_paths)
        })
        logger.info(f"  Result: True ISUP={isup} (label={label}) | Pred class={slide_pred}")

    if len(slide_results) < 3:
        logger.error("Too few slides successfully evaluated.")
        return

    # Compute metrics
    df_res = pd.DataFrame(slide_results)
    df_res.to_csv("storage/tcga_eval_results.csv", index=False)

    true_labels = df_res["label_true"].values
    pred_classes = df_res["slide_pred"].values

    pred_labels = []
    for pc in pred_classes:
        if pc == 0: pred_labels.append(0)
        elif pc == 1: pred_labels.append(1)
        elif pc == 4: pred_labels.append(2)
        elif pc == 5: pred_labels.append(3)
        else: pred_labels.append(1) # fallback

    pred_labels = np.array(pred_labels)

    qwk = cohen_kappa_score(true_labels, pred_labels, weights="quadratic")
    f1s = f1_score(true_labels, pred_labels, average=None, labels=[0,1,2,3], zero_division=0)

    print(f"\n=== REAL TCGA-PRAD EXTERNAL VALIDATION RESULTS ===")
    print(f"Slides evaluated: {eval_ok}")
    print(f"External QWK: {qwk:.4f}")
    print(f"F1 per label (Benign, ISUP1, ISUP4, ISUP5):", f1s)
    print()

    # Log to SQLite
    conn  = sqlite3.connect("db/cadx.db")
    cur  = conn.cursor()
    cur.execute("""CREATE TABLE IF NOT EXISTS experiments (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ts TEXT NOT NULL,
        experiment TEXT NOT NULL, metric_name TEXT NOT NULL,
        metric_value REAL NOT NULL)""")
    now = datetime.utcnow().isoformat()
    cur.execute("INSERT INTO experiments (ts, experiment, metric_name, metric_value) VALUES (?,?,?,?)",
                (now, "tcga_external_real", "external_qwk", qwk))
    cur.execute("INSERT INTO experiments (ts, experiment, metric_name, metric_value) VALUES (?,?,?,?)",
                (now, "tcga_external_real", "n_slides", eval_ok))
    for i, f1 in enumerate(f1s):
        cur.execute("INSERT INTO experiments (ts, experiment, metric_name, metric_value) VALUES (?,?,?,?)",
                    (now, "tcga_external_real", f"f1_label_{i}", f1))
    conn.commit(); conn.close()

    # Save results as JSON
    Path("storage/tcga_real_results.json").write_text(json.dumps({
        "qwk": float(qwk),
        "n_slides": int(eval_ok),
        "f1_0": float(f1s[0]),
        "f1_1": float(f1s[1]),
        "f1_2": float(f1s[2]),
        "f1_3": float(f1s[3])
    }, indent=2))

    # Confusion matrix plot
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_facecolor("#0d1117"); ax.set_facecolor("#0d1117")
    cm = confusion_matrix(true_labels, pred_labels, labels=[0,1,2,3])
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    tick_labels = ["ISUP 0", "ISUP 1", "ISUP 4", "ISUP 5"]
    ax.set_xticks(range(4)); ax.set_xticklabels(tick_labels, color="white")
    ax.set_yticks(range(4)); ax.set_yticklabels(tick_labels, color="white")
    ax.set_xlabel("Predicted", color="white"); ax.set_ylabel("True", color="white")
    ax.set_title(f"TCGA Real Confusion Matrix (QWK={qwk:.3f})", color="white", fontweight="bold")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                    color="black" if cm[i,j] > cm.max()*0.5 else "white")
    plt.tight_layout()
    plt.savefig("docs/assets/tcga_real_confusion_matrix.png", dpi=150, facecolor=fig.get_facecolor())
    plt.close()
    logger.info("Saved docs/assets/tcga_real_confusion_matrix.png")

if __name__ == "__main__":
    main()
