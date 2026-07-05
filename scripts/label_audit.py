"""
scripts/label_audit.py
Dumps a label mapping table and example tiles per grade for AUDIT.md.
Confirms ISUP grades 2 and 3 are absent in SICAPv2.
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

df = pd.read_csv("storage/manifest.csv")

print("=== SICAPv2 LABEL MAPPING TABLE ===")
print(f"{'tile_label':>12} | {'Gleason/ISUP':>20} | {'Tiles':>8} | {'Slides':>8}")
print("-" * 58)
label_map = {
    0: "NC  -> ISUP 0 (Benign)",
    1: "G3  -> ISUP 1 (3+3)",
    2: "G4  -> ISUP 4 (4+4)",
    3: "G5  -> ISUP 5 (5+5)",
}
for lbl, name in label_map.items():
    sub = df[df["tile_label"] == lbl]
    n_tiles = len(sub)
    n_slides = sub["slide_id"].nunique()
    print(f"{lbl:>12} | {name:>20} | {n_tiles:>8} | {n_slides:>8}")

print()
print("=== SAMPLE TILE PATHS PER GRADE ===")
for lbl, name in label_map.items():
    sub = df[df["tile_label"] == lbl]
    if len(sub) == 0:
        print(f"  tile_label={lbl} ({name}): NO TILES FOUND")
        continue
    for i in range(min(2, len(sub))):
        row = sub.iloc[i]
        pth = Path(row["tile_path"])
        size = pth.stat().st_size if pth.exists() else "MISSING"
        print(f"  [{name}] {pth.name} | {size} bytes | exists={pth.exists()}")

print()
print("=== GRADES PRESENT IN DATASET ===")
all_isup = sorted(df["slide_isup"].unique().tolist())
print(f"ISUP grades: {all_isup}")
print(f"ISUP 2 present: {2 in df['slide_isup'].values}")
print(f"ISUP 3 present: {3 in df['slide_isup'].values}")
print()

# Slide-level grade distribution
slides = df.groupby("slide_id")["slide_isup"].max()
print("=== SLIDE DISTRIBUTION BY ISUP GRADE ===")
print(slides.value_counts().sort_index().to_string())
