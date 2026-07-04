import streamlit as st
import numpy as np
import pandas as pd
import torch
from PIL import Image
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from lib.config import config
from lib.model import ProstateCADxModel
from lib.gradcam import GradCAM, apply_heatmap
from lib.db import db

st.set_page_config(
    page_title="Prostate Cancer CADx Gleason Grading",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Rich Aesthetics)
st.markdown("""
<style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: 'Inter', sans-serif;
    }
    .stHeader {
        background-color: #161b22;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #161b22;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #30363d;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

st.title("🔬 Prostate Cancer CADx (Gleason Grading)")
st.caption("Fine-tuned Convolutional Neural Network for Gleason pattern classification. Research-only assistive CADx tool.")

# Load model
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ProstateCADxModel(
        backbone=config.get("model.backbone", "resnet50"),
        num_classes=config.get("model.num_classes", 6),
        tile_classes=config.get("model.tile_classes", 4),
        aggregation=config.get("model.aggregation", "attention"),
        pretrained=False
    )
    checkpoint_path = Path("storage/checkpoints/best_model.pt")
    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model, device

model, device = load_model()

# Sidebar: Database status
st.sidebar.header("📊 CADx Loop Engine status")
try:
    skills = db.get_all_skills()
    done = [s for s in skills if s["status"] == "done"]
    st.sidebar.metric("Skills Implemented", f"{len(done)} / {len(skills)}")
    
    st.sidebar.subheader("Recent Runs")
    latest_metrics = db.get_latest_metrics()
    if latest_metrics:
        st.sidebar.write(f"**Val Loss**: {latest_metrics['val_loss']:.4f}")
        st.sidebar.write(f"**Val Kappa**: {latest_metrics['kappa']:.4f}")
        st.sidebar.write(f"**Epoch**: {latest_metrics['epoch']}")
except Exception:
    st.sidebar.warning("Database offline.")

col1, col2 = st.columns([1, 1])

with col1:
    st.header("Upload Histopathology Tile")
    uploaded_file = st.file_uploader("Upload slide image tile (JPG/PNG)", type=["jpg", "png", "jpeg"])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Slide Tile", use_column_width=True)
        
        # Preprocess & Predict
        if st.button("🔬 Run Gleason Analysis"):
            with st.spinner("Processing tissue activate maps..."):
                img_np = np.array(image)
                # Model inference
                img_tensor = torch.tensor(img_np.transpose(2, 0, 1) / 255.0, dtype=torch.float32).unsqueeze(0).to(device)
                
                # Grad-CAM Setup
                if config.get("model.backbone", "resnet50") == "resnet50":
                    target_layer = model.backbone.layer4[-1]
                else:
                    target_layer = model.backbone.features[-1]
                    
                gradcam = GradCAM(model, target_layer)
                
                # Calculate logits
                with torch.set_grad_enabled(True):
                    outputs = model(img_tensor)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    probs = torch.softmax(outputs, dim=1).cpu().detach().numpy()[0]
                    pred_class = np.argmax(probs)
                    
                    # Generate Grad-CAM map
                    heatmap = gradcam.generate_heatmap(img_tensor, class_idx=pred_class)
                    
                overlay = apply_heatmap(img_np, heatmap)
                
                # Show results in column 2
                with col2:
                    st.header("Diagnostic Prediction")
                    classes = ["Benign (Pattern 0)", "Gleason Pattern 3", "Gleason Pattern 4", "Gleason Pattern 5"]
                    
                    st.success(f"**Predicted Grade**: {classes[pred_class]}")
                    
                    # Probabilities bar chart
                    prob_df = pd.DataFrame({
                        "Pattern": classes,
                        "Probability": probs
                    })
                    st.bar_chart(prob_df.set_index("Pattern"))
                    
                    # Activation overlay
                    st.image(overlay, caption="Grad-CAM Activation Map (Red indicates visual focus)", use_column_width=True)
    else:
        st.info("Please upload a slide tile image to begin analysis.")
