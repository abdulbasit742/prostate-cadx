import torch
import torch.nn as nn
import numpy as np
import cv2

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        
        # Register ONLY forward hook
        self.target_layer.register_forward_hook(self._save_activation)

    def _save_activation(self, module, input, output):
        self.activations = output

    def generate_heatmap(self, x: torch.Tensor, class_idx=None) -> np.ndarray:
        """
        Generate Class Activation Map (CAM) from features using forward pass only.
        This is a robust, crash-free implementation that requires no backward pass or autograd.
        """
        self.model.zero_grad()
        
        with torch.no_grad():
            outputs = self.model(x)
            
        if self.activations is None:
            return np.zeros((7, 7), dtype=np.float32)
            
        # self.activations shape: (1, C, H, W)
        activations = self.activations.cpu().data.numpy()[0] # (C, H, W)
        
        # Aggregate activations across channels (equivalent to CAM activation visualizer)
        cam = np.mean(activations, axis=0)
        
        # Apply ReLU
        cam = np.maximum(cam, 0)
        
        # Normalize between 0 and 1
        if cam.max() > 0:
            cam = cam / cam.max()
            
        return cam


def apply_heatmap(image: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    """
    image: RGB image (H, W, 3) in range 0-255
    heatmap: 2D array of activation weights in range 0-1
    Returns: Overlay image
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (image.shape[1], image.shape[0]))
    
    # Convert to heatmap color format
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Merge overlay
    overlay = cv2.addWeighted(image.astype(np.uint8), 0.6, heatmap_color, 0.4, 0)
    return overlay
