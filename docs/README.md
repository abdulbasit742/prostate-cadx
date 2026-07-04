# Prostate Cancer CADx (Gleason Grading)

Prostate Cancer CADx is an end-to-end deep learning assistive system that processes histopathology Whole Slide Images (WSI), extracts tissue cores using Otsu segmentation, tiles them, normalizes stains using the Macenko algorithm, and trains a Multiple Instance Learning (MIL) model (ResNet50 / EfficientNet-B0 + Attention Pooling) to grade Gleason patterns and slide-level ISUP scores.

> [!WARNING]
> **Honesty Disclaimer & Guardrails**
> This project represents fine-tuned open CNN architectures for prostate Gleason grading on public histopathological datasets.
> **This is assistive CADx research and is NOT a clinically validated diagnostic tool.**
> It is intended exclusively for academic study and research benchmarking. No patient-identifying data leaves local storage.

## Project Structure

```
lib/         # Database, preprocessing, model, training, and evaluation libraries
scripts/     # Pipeline orchestrators, tiling, training, and Grad-CAM scripts
backend/     # Autonomy daemon, supervisor APIs, Streamlit interface, and scheduled tasks
skills/      # 100 modular, testable skill scripts
loop/        # Dependency engine and seeds configuration
tests/       # Synthetic pytest smoke tests
docs/        # RESULTS, MODEL_CARD, DATA, and RUNBOOK documentation
config/      # config.yaml and environment setup
```

## How to Run

1. Initialize the environment:
   ```bash
   py -3.11 -m venv venv
   .\venv\Scripts\pip install -r requirements.txt
   ```
2. Run smoke tests:
   ```bash
   .\venv\Scripts\pytest tests/
   ```
3. Run the end-to-end pipeline in smoke mode:
   ```bash
   .\venv\Scripts\python scripts/wire.py
   ```
4. Start the autonomous loop daemon:
   ```bash
   .\venv\Scripts\python backend/daemon.py
   ```
5. Launch the Streamlit visualization interface:
   ```bash
   .\venv\Scripts\streamlit run backend/streamlit_app.py
   ```
