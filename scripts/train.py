import os
import sys
import pandas as pd
import numpy as np
import torch
from pathlib import Path

# Inject project root into sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from lib.config import config
from lib.logging_setup import logger
from lib.db import db
from lib.data import GleasonDataset, get_augmentations
from lib.model import ProstateCADxModel
from lib.train import Trainer
from lib.gpu import gpu_monitor

def get_class_weights(df, label_col="tile_label", num_classes=4):
    counts = df[label_col].value_counts().to_dict()
    total = len(df)
    weights = []
    for c in range(num_classes):
        count = counts.get(c, 0)
        if count > 0:
            weights.append(total / (num_classes * count))
        else:
            weights.append(1.0)
    return weights

def run_training_session(batch_size, resume_checkpoint=None):
    manifest_path = Path(config.get("data.manifest_path", "storage/manifest.csv"))
    if not manifest_path.exists():
        logger.error("Tile manifest not found. Run tile_wsi.py first.")
        return False, None

    df = pd.read_csv(manifest_path)
    
    # 1. Stratified split on slide_id level (no patient leakage)
    slides = df.groupby("slide_id")["slide_isup"].max().reset_index()
    try:
        train_slides, val_slides = train_test_split(
            slides, 
            test_size=0.2, 
            stratify=slides["slide_isup"], 
            random_state=42
        )
    except ValueError:
        logger.warning("Stratified split failed (too few class members). Falling back to non-stratified split.")
        train_slides, val_slides = train_test_split(
            slides, 
            test_size=0.2, 
            random_state=42
        )
    
    train_df = df[df["slide_id"].isin(train_slides["slide_id"])]
    val_df = df[df["slide_id"].isin(val_slides["slide_id"])]
    
    # Log the split sizes per grade
    logger.info("Stratified split sizes per ISUP grade:")
    for grade in sorted(slides["slide_isup"].unique()):
        t_slides = len(train_slides[train_slides["slide_isup"] == grade])
        v_slides = len(val_slides[val_slides["slide_isup"] == grade])
        t_tiles = len(train_df[train_df["slide_isup"] == grade])
        v_tiles = len(val_df[val_df["slide_isup"] == grade])
        logger.info(f"Grade {grade} | Train Slides: {t_slides}, Val Slides: {v_slides} | Train Tiles: {t_tiles}, Val Tiles: {v_tiles}")
    
    # Create dataset format
    train_data = [{"image": r["tile_path"], "label": r["tile_label"]} for _, r in train_df.iterrows()]
    val_data = [{"image": r["tile_path"], "label": r["tile_label"]} for _, r in val_df.iterrows()]
    
    train_transform, val_transform = get_augmentations(config.get("data.tile_size", 256))
    
    # Enable Macenko stain normalization in dataset if configured
    stain_norm = config.get("data.stain_normalization", "macenko") == "macenko"
    
    train_dataset = GleasonDataset(train_data, transform=train_transform, normalize_stain=stain_norm)
    val_dataset = GleasonDataset(val_data, transform=val_transform, normalize_stain=stain_norm)
    
    # DataLoader configuration with pin_memory and prefetch
    num_workers = config.get("train.num_workers", os.cpu_count() - 1)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=max(1, num_workers), 
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=max(1, num_workers), 
        pin_memory=True,
        prefetch_factor=4 if num_workers > 0 else None
    )
    
    # Compute class weights
    class_weights = get_class_weights(train_df, "tile_label", num_classes=config.get("model.tile_classes", 4))
    
    # Model instantiation
    model = ProstateCADxModel(
        backbone=config.get("model.backbone", "resnet50"),
        num_classes=config.get("model.num_classes", 6),
        tile_classes=config.get("model.tile_classes", 4),
        aggregation=config.get("model.aggregation", "attention"),
        pretrained=True
    )
    
    trainer = Trainer(model, train_loader, val_loader, class_weights=class_weights,
                      resume_checkpoint=resume_checkpoint)
    
    try:
        best_kappa = trainer.fit()
        return True, best_kappa
    except RuntimeError as e:
        if "out of memory" in str(e):
            return False, None
        raise e

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Prostate Cancer CADx Train Script")
    parser.add_argument("--smoke", action="store_true", help="Run in smoke test mode")
    parser.add_argument("--lr", type=float, help="Learning rate override")
    parser.add_argument("--weight_decay", type=float, help="Weight decay override")
    parser.add_argument("--backbone", type=str, help="Backbone model override")
    parser.add_argument("--epochs", type=int, help="Epochs override")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    # Apply configuration overrides if specified
    if args.lr is not None:
        if "train" not in config.cfg: config.cfg["train"] = {}
        config.cfg["train"]["lr"] = args.lr
        logger.info(f"Overriding learning rate to {args.lr}")
        
    if args.weight_decay is not None:
        if "train" not in config.cfg: config.cfg["train"] = {}
        config.cfg["train"]["weight_decay"] = args.weight_decay
        logger.info(f"Overriding weight decay to {args.weight_decay}")
        
    if args.backbone is not None:
        if "model" not in config.cfg: config.cfg["model"] = {}
        config.cfg["model"]["backbone"] = args.backbone
        logger.info(f"Overriding model backbone to {args.backbone}")
        
    if args.epochs is not None:
        if "train" not in config.cfg: config.cfg["train"] = {}
        config.cfg["train"]["epochs"] = args.epochs
        logger.info(f"Overriding epochs to {args.epochs}")

    # Auto-tuning batch size loop
    start_batch_size = config.get("train.batch_size", 128)
    if args.smoke:
        logger.info("Running train.py in smoke mode with tiny batch size.")
        start_batch_size = 2
        if "train" not in config.cfg: config.cfg["train"] = {}
        config.cfg["train"]["epochs"] = 1
        
    batch_size = start_batch_size
    success = False
    best_kappa = 0.0
    
    while batch_size >= 2:
        logger.info(f"Attempting training with batch_size={batch_size}...")
        ok, kappa = run_training_session(batch_size, resume_checkpoint=args.resume)
        if ok:
            success = True
            best_kappa = kappa
            logger.info(f"Training session completed successfully at batch size {batch_size}. Validation Kappa: {best_kappa:.4f}")
            break
        else:
            logger.warning(f"OOM error at batch size {batch_size}. Halving batch size and retrying...")
            batch_size //= 2
            
    if not success:
        logger.error("Could not run training. VRAM exceeded even at batch size 2.")
        sys.exit(1)

if __name__ == "__main__":
    main()
