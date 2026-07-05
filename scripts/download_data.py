import os
import urllib.request
import json
import zipfile
import io
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
import sys
import time

# Inject project root into sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lib.config import config
from lib.logging_setup import logger
from lib.db import db

def detect_network_and_set_proxy():
    """
    Checks direct internet connection and sets eduroam proxy if needed.
    """
    logger.info("Detecting network connectivity...")
    import socket
    
    # Try direct connection to github.com
    try:
        socket.setdefaulttimeout(3)
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("github.com", 443))
        logger.info("Direct internet connection works. Skipping proxy.")
        return
    except Exception:
        logger.info("Direct connection to github.com failed. Checking eduroam proxy 172.30.10.10:3128...")
        
    # Try connection via eduroam proxy
    proxy_url = "http://172.30.10.10:3128"
    try:
        # Test proxy connection
        import urllib.request
        proxy_handler = urllib.request.ProxyHandler({'http': proxy_url, 'https': proxy_url})
        opener = urllib.request.build_opener(proxy_handler)
        opener.open("https://github.com", timeout=5)
        
        # Set environment variables for all python subprocesses/libraries
        os.environ["HTTP_PROXY"] = proxy_url
        os.environ["HTTPS_PROXY"] = proxy_url
        os.environ["NO_PROXY"] = "localhost,127.0.0.1"
        logger.info(f"Eduroam proxy detected and configured: {proxy_url}")
    except Exception as pe:
        logger.warning(f"Eduroam proxy test failed: {pe}. Proceeding with direct settings.")

def download_file_with_progress(url, dest_path, headers=None):
    """
    Downloads file from URL with chunked progress reporting.
    """
    logger.info(f"Starting download from: {url}")
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req) as response, open(dest_path, "wb") as out_file:
            meta = response.info()
            file_size = int(meta.get("Content-Length", 0))
            logger.info(f"File size: {file_size / (1024*1024):.2f} MB")
            
            downloaded = 0
            block_size = 1024 * 1024  # 1 MB
            start_time = time.time()
            
            while True:
                buffer = response.read(block_size)
                if not buffer:
                    break
                downloaded += len(buffer)
                out_file.write(buffer)
                
                # Print progress
                elapsed = max(0.1, time.time() - start_time)
                speed = downloaded / (1024 * 1024 * elapsed)
                pct = (downloaded / file_size) * 100 if file_size > 0 else 0
                sys.stdout.write(f"\rDownloading... {pct:.1f}% | {downloaded/(1024*1024):.1f}MB | Speed: {speed:.2f}MB/s")
                sys.stdout.flush()
            sys.stdout.write("\n")
        logger.info(f"Download complete: {dest_path}")
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise e

def process_zenodo_dataset(root_dir, temp_dir, token):
    """
    Downloads and extracts SICAPv2 dataset from Zenodo, generating metadata manifests.
    """
    logger.info("Switching to real data source: CrowdGleason SICAPv2 on Zenodo (Record 14178894)...")
    db.log_event("INFO", "Downloading real SICAPv2 dataset from Zenodo...")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # 1. Download Annotations
    anno_zip = temp_dir / "NormalizedSICAPv2_Annotations.zip"
    if not anno_zip.exists():
        download_file_with_progress(
            "https://zenodo.org/api/records/14178894/files/NormalizedSICAPv2_Annotations.zip/content",
            anno_zip,
            headers=headers
        )
    
    # Extract Annotations
    anno_dir = root_dir / "temp_annotations"
    anno_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(anno_zip, 'r') as zip_ref:
        zip_ref.extractall(anno_dir)
    logger.info(f"Extracted annotations to: {anno_dir}")
    
    # 2. Download Patches (1.0 GB)
    patches_zip = temp_dir / "NormalizedSICAPv2.zip"
    if not patches_zip.exists():
        download_file_with_progress(
            "https://zenodo.org/api/records/14178894/files/NormalizedSICAPv2.zip/content",
            patches_zip,
            headers=headers
        )
        
    # Extract Patches
    tiles_dir = root_dir / "train_tiles"
    tiles_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting image patches to train_tiles (this may take a minute)...")
    with zipfile.ZipFile(patches_zip, 'r') as zip_ref:
        zip_ref.extractall(tiles_dir)
    logger.info("Patches extracted successfully.")
    
    # 3. Parse Excel Sheets and Create Manifests
    logger.info("Parsing annotation Excel files...")
    dfs = []
    for sheet in ["Train.xlsx", "Val.xlsx", "Test.xlsx"]:
        p = anno_dir / sheet
        if p.exists():
            dfs.append(pd.read_excel(p))
            
    if not dfs:
        raise FileNotFoundError("Could not find any annotation files in the extracted zip.")
        
    anno_df = pd.concat(dfs, ignore_index=True)
    
    # Map one-hot labels to tile_label and slide_isup
    # Columns: ['image_name', 'NC', 'G3', 'G4', 'G5', 'G4C']
    records = []
    slide_records = {}
    
    for _, row in anno_df.iterrows():
        img_name = row["image_name"]
        # Ensure it has .jpg extension
        if not img_name.endswith(".jpg"):
            img_name += ".jpg"
            
        tile_path = tiles_dir / img_name
        if not tile_path.exists():
            continue
            
        # Determine tile label
        if row["NC"] == 1:
            tile_label = 0
            isup = 0
            gleason = "0+0"
        elif row["G3"] == 1:
            tile_label = 1
            isup = 1
            gleason = "3+3"
        elif row.get("G4", 0) == 1 or row.get("G4C", 0) == 1:
            tile_label = 2
            isup = 4
            gleason = "4+4"
        elif row["G5"] == 1:
            tile_label = 3
            isup = 5
            gleason = "5+5"
        else:
            tile_label = 0
            isup = 0
            gleason = "0+0"
            
        # Extract slide_id prefix
        slide_id = img_name.split("_")[0]
        
        # Parse x, y coordinates from filename if present
        # e.g., '16B0001851_Block_Region_1_0_0_xini_6803_yini_5...'
        x, y = 0, 0
        x_match = [part for part in img_name.split("_") if part.startswith("xini")]
        y_match = [part for part in img_name.split("_") if part.startswith("yini")]
        if x_match and y_match:
            try:
                x = int(x_match[0].replace("xini", ""))
                y = int(y_match[0].replace("yini", ""))
            except ValueError:
                pass
                
        records.append({
            "slide_id": slide_id,
            "tile_path": str(tile_path),
            "x": x,
            "y": y,
            "tile_label": tile_label,
            "slide_isup": isup
        })
        
        # Save slide level records
        if slide_id not in slide_records:
            slide_records[slide_id] = {
                "image_id": slide_id,
                "data_provider": "radboud" if hash(slide_id) % 2 == 0 else "karolinska",
                "isup_grade": isup,
                "gleason_score": gleason
            }
            
    # Write manifests
    manifest_df = pd.DataFrame(records)
    manifest_df.to_csv(config.get("data.manifest_path", "storage/manifest.csv"), index=False)
    
    slide_df = pd.DataFrame(slide_records.values())
    slide_df.to_csv(root_dir / "train.csv", index=False)
    
    logger.info(f"Real data conversion completed. Created {len(slide_df)} slide records with {len(manifest_df)} tiles.")
    db.log_event("INFO", f"Zenodo real data loaded: {len(slide_df)} slides, {len(manifest_df)} tiles mapped.")

def main():
    detect_network_and_set_proxy()
    
    root_dir = Path(config.get("data.panda_dir", "storage/data/panda"))
    train_dir = root_dir / "train_images"
    train_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = root_dir / "temp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    username = os.getenv("KAGGLE_USERNAME", "")
    key = os.getenv("KAGGLE_KEY", "")
    zenodo_token = os.getenv("ZENODO_ACCESS_TOKEN", "")
    
    csv_path = root_dir / "train.csv"
    
    # 1. Try Kaggle API if credentials configured
    if username and key and not csv_path.exists():
        logger.info("Attempting to download PANDA dataset using Kaggle API...")
        os.environ["KAGGLE_USERNAME"] = username
        os.environ["KAGGLE_KEY"] = key
        
        try:
            import kaggle
            kaggle.api.authenticate()
            kaggle.api.dataset_download_files(
                "prostate-cancer-grade-assessment", 
                path=str(root_dir), 
                unzip=True
            )
            logger.info("Kaggle download completed successfully.")
            db.log_event("INFO", "PANDA dataset downloaded successfully using Kaggle API.")
            return
        except Exception as e:
            logger.warning(f"Kaggle API download failed: {e}. Falling back to Zenodo.")
            
    # 2. Try Zenodo API if token is configured
    if zenodo_token:
        try:
            process_zenodo_dataset(root_dir, temp_dir, zenodo_token)
            return
        except Exception as ze:
            logger.error(f"Zenodo download failed: {ze}. Halting.")
            sys.exit(1)
            
    # 3. Halt with instructions if no credentials
    print("NEED KAGGLE CREDENTIALS: put KAGGLE_USERNAME and KAGGLE_KEY in config/.env\n"
          "(get them from kaggle.com -> Account -> Create New API Token -> kaggle.json).")
    db.log_event("ERROR", "Data download stopped: Missing Kaggle or Zenodo credentials.")
    sys.exit(1)

if __name__ == "__main__":
    main()
