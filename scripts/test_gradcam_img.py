import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("Step 1: Imports done.")
import torch
import numpy as np
import pandas as pd
from PIL import Image
import cv2
from lib.config import config
from lib.model import ProstateCADxModel
from lib.gradcam import GradCAM, apply_heatmap

print("Step 2: Model init starting.")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ProstateCADxModel(
    backbone="resnet50",
    num_classes=6,
    tile_classes=4,
    aggregation="attention",
    pretrained=False
)
print("Step 3: Loading state dict.")
best_model_path = Path("storage/checkpoints/best_model.pt")
if best_model_path.exists():
    model.load_state_dict(torch.load(best_model_path, map_location=device))
model = model.to(device)
model.eval()

print("Step 4: Creating GradCAM object.")
target_layer = model.backbone.layer4[-1]
gradcam = GradCAM(model, target_layer)

print("Step 5: Loading image from manifest.")
manifest_path = Path("storage/manifest.csv")
df = pd.read_csv(manifest_path)
sample_img_path = Path(df.iloc[0]["tile_path"])
print("Image path:", sample_img_path)

img = Image.open(sample_img_path).convert("RGB")
img_np = np.array(img)
print("Image loaded, shape:", img_np.shape)

print("Step 6: Running forward pass to populate activations.")
img_tensor = torch.tensor(img_np.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
heatmap = gradcam.generate_heatmap(img_tensor, class_idx=1)
print("Heatmap computed, shape:", heatmap.shape)

print("Step 7: Applying heatmap overlay.")
overlay = apply_heatmap(img_np, heatmap)
print("Overlay applied.")

print("Step 8: Saving overlay image.")
output_dir = Path("docs/assets")
output_dir.mkdir(parents=True, exist_ok=True)
overlay_path = output_dir / "gradcam_sample.jpg"
Image.fromarray(overlay).save(overlay_path)
print("Overlay saved.")

print("ALL STEPS COMPLETED.")
