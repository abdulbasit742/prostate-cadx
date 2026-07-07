import time, torch, torch.nn as nn
import pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from lib.model import ProstateCADxModel
from lib.data import GleasonDataset, get_augmentations

def run_prof():
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", DEVICE)

    df = pd.read_csv("storage/manifest.csv")
    slides = df.groupby("slide_id")["slide_isup"].max().reset_index()
    train_slides, _ = train_test_split(slides, test_size=0.2, stratify=slides["slide_isup"], random_state=42)
    train_df = df[df["slide_id"].isin(train_slides["slide_id"])]
    records = [{"image": r["tile_path"], "label": r["tile_label"]} for _, r in train_df.iterrows()]

    records = records[:100]
    print("Preloading 100 records...")
    train_tf, _ = get_augmentations(256)
    t0 = time.time()
    ds = GleasonDataset(records, transform=train_tf)
    print(f"Preloaded in {time.time() - t0:.2f}s")

    loader = DataLoader(ds, batch_size=32, shuffle=True, num_workers=0)
    t0 = time.time()
    for imgs, targets in loader:
        pass
    print(f"DataLoader iteration: {time.time() - t0:.2f}s")

    model = ProstateCADxModel(backbone="resnet50", num_classes=6, tile_classes=4, aggregation="attention", pretrained=False).to(DEVICE)
    t0 = time.time()
    for imgs, targets in loader:
        imgs = imgs.to(DEVICE)
        out = model(imgs)
    print(f"DataLoader + Model forward: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    try:
        run_prof()
    except Exception as e:
        import traceback
        traceback.print_exc()
