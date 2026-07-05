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
from lib.eval import Evaluator

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    manifest_path = Path(config.get("data.manifest_path", "storage/manifest.csv"))
    if not manifest_path.exists():
        logger.error("Tile manifest not found. Run tile_wsi.py first.")
        return

    df = pd.read_csv(manifest_path)
    slides = df.groupby("slide_id")["slide_isup"].max().reset_index()
    try:
        _, val_slides = train_test_split(
            slides, 
            test_size=0.2, 
            stratify=slides["slide_isup"], 
            random_state=42
        )
    except ValueError:
        logger.warning("Stratified split failed (too few class members). Falling back to non-stratified split.")
        _, val_slides = train_test_split(
            slides, 
            test_size=0.2, 
            random_state=42
        )
    val_df = df[df["slide_id"].isin(val_slides["slide_id"])]
    
    val_data = [{"image": r["tile_path"], "label": r["tile_label"]} for _, r in val_df.iterrows()]
    _, val_transform = get_augmentations(config.get("data.tile_size", 256))
    
    val_dataset = GleasonDataset(val_data, transform=val_transform)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    # Load model
    model = ProstateCADxModel(
        backbone=config.get("model.backbone", "resnet50"),
        num_classes=config.get("model.num_classes", 6),
        tile_classes=config.get("model.tile_classes", 4),
        aggregation=config.get("model.aggregation", "attention"),
        pretrained=False
    )
    
    best_model_path = Path(config.get("model.checkpoint_dir", "storage/checkpoints")) / "best_model.pt"
    
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path, map_location=device))
        logger.info(f"Loaded best model parameters from {best_model_path}")
    else:
        logger.warning(f"No checkpoint found at {best_model_path}. Using uninitialized weights.")
        
    model = model.to(device)
    model.eval()
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)
            all_preds.extend(preds)
            all_targets.extend(targets.numpy())
            all_probs.extend(probs)
            
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    all_probs = np.array(all_probs)
    
    evaluator = Evaluator()
    metrics = evaluator.compute_metrics(all_targets, all_preds, all_probs, num_classes=config.get("model.tile_classes", 4))
    cm_plot = evaluator.plot_confusion_matrix(all_targets, all_preds, classes=["benign", "gleason_3", "gleason_4", "gleason_5"])
    
    # Save results to DB and files
    db.log_event("INFO", f"Slide evaluation completed. Validation Kappa: {metrics['qwk']:.4f}")
    
    # Write RESULTS.md file content
    results_path = Path("docs/RESULTS.md")
    with open(results_path, "w") as f:
        f.write(f"# Prostate Cancer Gleason Grading Evaluation Results\n\n")
        f.write(f"Generated at: {pd.Timestamp.now()}\n\n")
        f.write(f"## Clinical Validation Summary\n\n")
        f.write(f"- **Quadratic Weighted Kappa (QWK)**: {metrics['qwk']:.4f}\n")
        f.write(f"- **Accuracy**: {metrics['report']['accuracy']:.4f}\n\n")
        f.write(f"### Classification Metrics per Class\n\n")
        f.write(f"| Class | Precision | Recall | F1-Score |\n")
        f.write(f"| --- | --- | --- | --- |\n")
        for cls_name, cls_metrics in metrics['report'].items():
            if isinstance(cls_metrics, dict):
                f.write(f"| {cls_name} | {cls_metrics['precision']:.4f} | {cls_metrics['recall']:.4f} | {cls_metrics['f1-score']:.4f} |\n")
        f.write(f"\n## Visualizations\n\n")
        f.write(f"### Confusion Matrix\n")
        f.write(f"![Confusion Matrix](assets/confusion_matrix.png)\n")
        
    logger.info(f"Results report successfully generated at {results_path}")

if __name__ == "__main__":
    main()
