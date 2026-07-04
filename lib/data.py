import os
import numpy as np
import pandas as pd
import cv2
import tiffslide as ts
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from lib.logging_setup import logger
from lib.config import config

class MacenkoNormalizer:
    """
    Macenko Stain Normalization in pure NumPy.
    Ref: Macenko et al. (2009) "A method for normalizing histology slides..."
    """
    def __init__(self):
        # Reference values (stain matrices for H&E)
        self.Io = 255.0
        self.beta = 0.15
        self.alpha = 1.0
        self.HERef = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581]
        ])
        self.maxCRef = np.array([1.9705, 1.0308])

    def normalize(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize H&E image.
        """
        img_np = np.array(img).astype(np.float64)
        h, w, c = img_np.shape
        img_vec = img_np.reshape((-1, 3))

        # Optical density (OD)
        OD = -np.log((img_vec + 1.0) / self.Io)
        ODhat = OD[np.all(OD >= self.beta, axis=1)]
        
        if ODhat.shape[0] == 0:
            return img # No tissue or extremely light

        # SVD on OD covariance
        eigvals, eigvecs = np.linalg.eigh(np.cov(ODhat.T))
        
        # Project data onto plane spanned by 2 largest eigenvectors
        best_2_eigvecs = eigvecs[:, [1, 2]]
        
        # Project and normalize
        T = np.dot(ODhat, best_2_eigvecs)
        phi = np.arctan2(T[:, 1], T[:, 0])
        
        min_phi = np.percentile(phi, self.alpha)
        max_phi = np.percentile(phi, 100.0 - self.alpha)
        
        v_min = np.dot(best_2_eigvecs, np.array([np.cos(min_phi), np.sin(min_phi)]))
        v_max = np.dot(best_2_eigvecs, np.array([np.cos(max_phi), np.sin(max_phi)]))
        
        # Check directions
        if v_min[0] < 0:
            v_min *= -1
        if v_max[0] < 0:
            v_max *= -1
            
        HE = np.array([v_min, v_max]).T
        
        # Project OD back
        C = np.linalg.lstsq(HE, OD.T, rcond=None)[0]
        
        # Normalize
        maxC = np.percentile(C, 99.0, axis=1)
        C = (C.T / maxC).T
        C = (C.T * self.maxCRef).T
        
        # Reconstruct image
        img_norm = self.Io * np.exp(-np.dot(self.HERef, C))
        img_norm = np.clip(img_norm.T, 0, 255).astype(np.uint8)
        return img_norm.reshape((h, w, c))


class WSITiler:
    def __init__(self, tile_size=256, min_tissue_ratio=0.3):
        self.tile_size = tile_size
        self.min_tissue_ratio = min_tissue_ratio
        self.normalizer = MacenkoNormalizer()

    def get_tissue_mask(self, slide_img: Image.Image) -> np.ndarray:
        """
        Produce tissue mask using Otsu threshold on grayscale slide thumbnail.
        """
        img_np = np.array(slide_img)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Otsu thresholding
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological opening/closing to clean noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        return mask

    def tile_slide(self, slide_path: str, is_smoke=False) -> list:
        """
        Grid tile the slide, filtering out background and blurry tiles.
        """
        tiles = []
        if is_smoke or not Path(slide_path).exists():
            # Generate fake tiles for smoke mode
            for i in range(16):
                fake_tile = np.random.randint(180, 255, (self.tile_size, self.tile_size, 3), dtype=np.uint8)
                tiles.append({
                    "tile": fake_tile,
                    "x": i * self.tile_size,
                    "y": 0,
                    "tissue_ratio": 1.0
                })
            return tiles

        try:
            slide = ts.TiffSlide(slide_path)
            # Read at thumbnail level to detect tissue mask
            thumb_size = (slide.dimensions[0] // 32, slide.dimensions[1] // 32)
            thumb = slide.get_thumbnail(thumb_size)
            mask = self.get_tissue_mask(thumb)
            
            # Map coordinates and slide tiles
            scale_x = slide.dimensions[0] / thumb_size[0]
            scale_y = slide.dimensions[1] / thumb_size[1]
            
            # Retrieve tiles in grids
            for y in range(0, slide.dimensions[1], self.tile_size):
                for x in range(0, slide.dimensions[0], self.tile_size):
                    # Check overlap with tissue mask
                    mx = int(x / scale_x)
                    my = int(y / scale_y)
                    mw = max(1, int(self.tile_size / scale_x))
                    mh = max(1, int(self.tile_size / scale_y))
                    
                    mask_roi = mask[my:my+mh, mx:mx+mw]
                    if mask_roi.size > 0:
                        tissue_ratio = np.mean(mask_roi == 255)
                        if tissue_ratio >= self.min_tissue_ratio:
                            # Extract tile image
                            tile_img = slide.read_region((x, y), 0, (self.tile_size, self.tile_size))
                            tile_np = np.array(tile_img.convert("RGB"))
                            
                            # Clean blur via Laplacian variance
                            lap = cv2.Laplacian(tile_np, cv2.CV_64F).var()
                            if lap > 10.0:  # Simple blur filter threshold
                                tiles.append({
                                    "tile": tile_np,
                                    "x": x,
                                    "y": y,
                                    "tissue_ratio": tissue_ratio
                                })
        except Exception as e:
            logger.error(f"Error tiling slide {slide_path}: {e}")
            
        return tiles


class GleasonDataset(Dataset):
    def __init__(self, data_list, transform=None, normalize_stain=False):
        """
        data_list: list of dicts with {"image": np.ndarray or str, "label": int}
        """
        self.transform = transform
        self.normalize_stain = normalize_stain
        self.stain_norm = MacenkoNormalizer()
        self.data_list = []
        
        logger.info(f"Preloading {len(data_list)} tiles into system RAM...")
        for item in data_list:
            img = item["image"]
            if isinstance(img, (str, Path)):
                try:
                    img_np = np.array(Image.open(str(img)).convert("RGB"))
                    self.data_list.append({"image": img_np, "label": item["label"]})
                except Exception as e:
                    logger.warning(f"Failed to preload tile {img}: {e}")
            else:
                self.data_list.append(item)
        logger.info(f"Preloaded {len(self.data_list)} tiles successfully.")

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        item = self.data_list[idx]
        img = item["image"]
            
        if self.normalize_stain:
            try:
                img = self.stain_norm.normalize(img)
            except Exception:
                pass
                
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented["image"]
        else:
            img = img.transpose(2, 0, 1) / 255.0 # CHW
            img = torch.tensor(img, dtype=torch.float32)
            
        label = item["label"]
        return img, torch.tensor(label, dtype=torch.long)


def get_augmentations(tile_size=256):
    train_transform = A.Compose([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    val_transform = A.Compose([
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    return train_transform, val_transform
