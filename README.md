# ATLAS-Net: An Adaptive Tabular-guided Layerwise Alignment and Synthesis Network for Multimodal Neuroimaging and Clinical Phenotype Fusion

[![Paper](https://img.shields.io/badge/Paper-KDD2026-red)]() [![Python](https://img.shields.io/badge/Python-3.9-blue)]() [![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange)]()

## News
🎉 Our paper has been accepted by **KDD 2026** (The 32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining)!

## Overview
This repository provides a complete implementation for **AD vs. HC (Alzheimer's Disease vs. Healthy Control) classification** using multi-modal MRI data (sMRI + fMRI). The model integrates ResNet-based CNN, Vision Transformer, HyperNetwork for tabular-conditioned adaptive parameters, and cross-modal attention for feature fusion.

**For other classification tasks (Bipolar Disorder, Schizophrenia, MDD, etc), simply replace the dataset path.**

## Architecture
![ATLAS-Net Architecture](https://raw.githubusercontent.com/liangchuang11/ATLAS-Net/main/Architecture.jpg)

## Quick Start
```bash
# Install dependencies
pip install torch numpy scikit-learn einops

# Update dataset path in ATLAS-Net.py
cached_path = "/path/to/your/dataset.pt"

# Run 
python ATLAS-Net.py
