import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

print("Step 1: Imports done.")
import torch
import numpy as np
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

print("Step 5: Creating dummy image.")
img_np = np.random.randint(180, 255, (256, 256, 3), dtype=np.uint8)
img_tensor = torch.tensor(img_np.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)

print("Step 6: Running forward pass to populate activations.")
model.zero_grad()
outputs = model(img_tensor)
if isinstance(outputs, tuple):
    outputs = outputs[0]
print("Forward pass outputs shape:", outputs.shape)

print("Step 7: Selecting class and computing score.")
class_idx = torch.argmax(outputs, dim=1).item()
score = outputs[0, class_idx]
print(f"Selected class: {class_idx}, Score: {score.item()}")

print("Step 8: Computing autograd gradients.")
# Make sure activations requires_grad is true
gradcam.activations.requires_grad_(True)
print("Activations requires_grad state:", gradcam.activations.requires_grad)

try:
    grads = torch.autograd.grad(score, gradcam.activations, retain_graph=True)[0]
    print("Grads shape:", grads.shape)
except Exception as e:
    print("Autograd grad failed:", e)

print("Step 9: Global average pooling.")
gradients = grads.cpu().data.numpy()[0]
activations = gradcam.activations.cpu().data.numpy()[0]
weights = np.mean(gradients, axis=(1, 2))

print("Step 10: Computing heatmap sum.")
cam = np.zeros(activations.shape[1:], dtype=np.float32)
for i, w in enumerate(weights):
    cam += w * activations[i]
cam = np.maximum(cam, 0)
if cam.max() > 0:
    cam = cam / cam.max()
print("Cam max:", cam.max())

print("Step 11: Applying heatmap.")
overlay = apply_heatmap(img_np, cam)
print("Overlay shape:", overlay.shape)

print("ALL STEPS COMPLETED.")
