import pytest
import torch
import numpy as np
from pathlib import Path
from lib.model import ProstateCADxModel
from lib.db import Database
from lib.gradcam import GradCAM

def test_model_forward():
    model = ProstateCADxModel(
        backbone="resnet50",
        num_classes=6,
        tile_classes=4,
        aggregation="attention",
        pretrained=False
    )
    # Single tile input (batch_size=2, C=3, H=256, W=256)
    x = torch.randn(2, 3, 256, 256)
    logits = model(x)
    assert logits.shape == (2, 4)

def test_model_aggregation():
    model = ProstateCADxModel(
        backbone="resnet50",
        num_classes=6,
        tile_classes=4,
        aggregation="attention",
        pretrained=False
    )
    # Bag input (batch_size=2, num_tiles=5, C=3, H=256, W=256)
    x = torch.randn(2, 5, 3, 256, 256)
    slide_logits, tile_logits = model(x)
    assert slide_logits.shape == (2, 6)
    assert tile_logits.shape == (2, 5, 4)

def test_database():
    temp_db_path = "storage/test_cadx.db"
    db = Database(db_path=temp_db_path)
    db.register_skill(1, "test_skill", "A", "pending", [])
    
    skills = db.get_pending_skills()
    assert len(skills) == 1
    assert skills[0]["name"] == "test_skill"
    
    db.update_skill_status(1, "done")
    assert len(db.get_pending_skills()) == 0
    
    # Cleanup
    if Path(temp_db_path).exists():
        os = None
        try:
            import os
            os.remove(temp_db_path)
        except Exception:
            pass

def test_gradcam():
    model = ProstateCADxModel(
        backbone="resnet50",
        num_classes=6,
        tile_classes=4,
        aggregation="attention",
        pretrained=False
    )
    target_layer = model.backbone.layer4[-1]
    gradcam = GradCAM(model, target_layer)
    x = torch.randn(1, 3, 256, 256)
    heatmap = gradcam.generate_heatmap(x, class_idx=1)
    assert heatmap.shape == (7, 7) or heatmap.shape == (8, 8) # Depends on output shape of layer4
    assert np.max(heatmap) <= 1.0
