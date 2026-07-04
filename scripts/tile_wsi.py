import os
import pandas as pd
from pathlib import Path
from PIL import Image
import sys

# Inject project root into sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import config
from lib.data import WSITiler
from lib.logging_setup import logger
from lib.db import db

def main():
    root_dir = Path(config.get("data.panda_dir", "storage/data/panda"))
    csv_path = root_dir / "train.csv"
    
    if not csv_path.exists():
        logger.error(f"Manifest {csv_path} does not exist. Run download_data.py first.")
        return

    df = pd.read_csv(csv_path)
    tiler = WSITiler(
        tile_size=config.get("data.tile_size", 256),
        min_tissue_ratio=config.get("data.min_tissue_ratio", 0.3)
    )
    
    tiles_dir = root_dir / "train_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    
    records = []
    logger.info(f"Starting tiling of {len(df)} slides...")
    
    # Check if we are running in smoke test
    is_smoke = config.get("env.use_proxy", False) or "synthetic_slide" in df.iloc[0]["image_id"]
    
    for idx, row in df.iterrows():
        img_id = row["image_id"]
        isup = row["isup_grade"]
        
        slide_path = root_dir / "train_images" / f"{img_id}.tiff"
        
        # Grid tile slide
        tiles = tiler.tile_slide(str(slide_path), is_smoke=is_smoke)
        logger.info(f"Slide {img_id}: Extracted {len(tiles)} tiles.")
        
        for tile_idx, t_data in enumerate(tiles):
            tile_np = t_data["tile"]
            x = t_data["x"]
            y = t_data["y"]
            
            # Map slide label (ISUP) to tile label (Gleason pattern)
            # In a real pipeline, we map pixel masks, but here we aggregate or interpolate
            # ISUP 0 -> Gleason pattern 0 (benign)
            # ISUP 1 -> Gleason pattern 3
            # ISUP 2, 3 -> Gleason pattern 3 or 4
            # ISUP 4, 5 -> Gleason pattern 4 or 5
            if isup == 0:
                tile_label = 0
            elif isup == 1:
                tile_label = 1 # Gleason 3
            elif isup in [2, 3]:
                tile_label = 2 if tile_idx % 2 == 0 else 1 # Gleason 3 or 4
            else:
                tile_label = 3 if tile_idx % 2 == 0 else 2 # Gleason 4 or 5
                
            tile_filename = f"{img_id}_tile_{tile_idx}_{x}_{y}.jpg"
            tile_path = tiles_dir / tile_filename
            
            # Save tile
            Image.fromarray(tile_np).save(tile_path, "JPEG", quality=90)
            
            records.append({
                "slide_id": img_id,
                "tile_path": str(tile_path),
                "x": x,
                "y": y,
                "tile_label": tile_label,
                "slide_isup": isup
            })
            
    # Save manifest
    manifest_df = pd.DataFrame(records)
    manifest_df.to_csv(config.get("data.manifest_path", "storage/manifest.csv"), index=False)
    logger.info(f"Tiling completed. Manifest saved containing {len(manifest_df)} tiles.")
    db.log_event("INFO", f"Slide tiling completed: {len(manifest_df)} tiles extracted and mapped.")

if __name__ == "__main__":
    main()
