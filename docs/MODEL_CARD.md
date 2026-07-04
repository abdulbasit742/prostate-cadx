# Model Card - Prostate Cancer Gleason Grading

## Model Details
- **Developer**: abdulbasit742
- **Model Type**: ResNet50 / EfficientNet-B0 + Attention Pooling (MIL)
- **Task**: Tile-level Gleason Pattern Classification (0, 3, 4, 5) and Slide-level ISUP Grade Prediction (0 to 5).
- **Stain Normalization**: Macenko method.

## Intended Use
- **Primary Use Case**: Assistive computer-aided diagnostics (CADx) research for pathology education.
- **Out of Scope**: Clinical decision support, direct diagnostic decisions, or any application without a certified human-in-the-loop pathologist.

## Limitations
- Model performance is highly dependent on stain consistency; stain normalization (Macenko) is applied but site-specific artifacts can still cause prediction drift.
- Biopsy cores contain sectioning artifacts, air bubbles, and wrinkles which may trigger false activations.
- Currently trained on public datasets with limited demographic metadata.
