# Prostate Cancer CADx Datasets

This document catalogs the public histopathology datasets supported by this CADx system.

## 1. PANDA Challenge (Primary)
- **Source**: Kaggle (prostate-cancer-grade-assessment)
- **Description**: Largest public dataset for prostate cancer grading. Contains ~11,000 WSI (Whole Slide Images) with slide-level ISUP grades (0 to 5) and pixel-level Gleason mask patterns.
- **Usage**: Used as the primary training and internal validation dataset.
- **License**: Research-use only.

## 2. Gleason 2019 Challenge (Secondary)
- **Source**: Grand Challenge
- **Description**: Tissue Microarrays (TMA) containing pixel-level Gleason annotation masks from multiple expert pathologists.
- **Usage**: Detailed tile-level semantic validation.
- **License**: Open access for research.

## 3. TCGA-PRAD (External Validation)
- **Source**: GDC (Genomic Data Commons) Portal
- **Description**: Whole Slide Images and clinical labels for Prostate Adenocarcinoma.
- **Usage**: Used for zero-shot external validation to measure model generalization against stain and site variance.
- **License**: NIH public domain.
