# Pixel Play '26: Video Anomaly Detection

[![Kaggle Competition](https://img.shields.io/badge/Kaggle-Competition-blue)](https://www.kaggle.com/competitions/pixel-play-26)
[![Score](https://img.shields.io/badge/Final%20Score-0.6310%20AP-green)](https://www.kaggle.com/competitions/pixel-play-26)
[![Python](https://img.shields.io/badge/Python-3.8+-yellow)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)

## Competition Overview

This repository contains my solution for the **Pixel Play '26** Kaggle competition, which focuses on detecting anomalous events in surveillance footage from the Avenue dataset (corrupted version).

**Final Score: 0.6310 Average Precision** (54% improvement from baseline)

### Problem Statement
- **Task:** Unsupervised anomaly detection in video surveillance
- **Training Data:** Videos containing only "normal" behavior
- **Test Data:** Videos with both normal and anomalous frames
- **Metric:** Average Precision (AP)

---

## Repository Structure

```
pixel-play-26/
│
├── notebooks/
│   ├── Phase1_Exp1_SimpleConvAE_Baseline.ipynb
│   ├── Phase1_Exp2_UNet_SkipConnections.ipynb
│   ├── Phase2_Exp1_MAX_Error_Scoring.ipynb
│   ├── Phase2_Exp2_Ensemble.ipynb
│   ├── Phase3_Exp1_3DCNN.ipynb
│   ├── Phase3_Exp2_ConvLSTM.ipynb
│   ├── Phase3_Exp3_OpticalFlow.ipynb
│   ├── Phase4_Exp1_TemporalSmoothing.ipynb
│   ├── Phase4_Exp2_AlternativeSmoothing.ipynb
│   └── Phase5_Exp1_MultiScale.ipynb
│
├── submissions/
│   └── (CSV submission files)
│
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## Solution Journey

### Score Progression

| Phase | Approach | Score | Improvement |
|-------|----------|-------|-------------|
| 1 | Simple ConvAE + Mean Error | 0.41 | Baseline |
| 1 | U-Net with Skip Connections | 0.33 | -20% |
| 2 | Simple ConvAE + MAX Error | 0.53 | +29% |
| 2 | 5-Model Ensemble | 0.5344 | +0.8% |
| 3 | 3D CNN | 0.53 | 0% |
| 3 | ConvLSTM Prediction | 0.41-0.45 | -15% |
| 3 | Optical Flow | 0.31 | -24% |
| 4 | ConvAE + Gaussian Smoothing (σ=3) | 0.6041 | +13% |
| 5 | **Multi-Scale + MAX Fusion** | **0.6310** | **+4.5%** |

---

## Key Discoveries

### 1. MAX Error >> Mean Error (+29%)
```python
# Instead of mean error:
# score = torch.mean((input - recon) ** 2)

# Use MAX error for localized anomalies:
score = torch.amax((input - recon) ** 2, dim=(1, 2, 3))
```
**Why:** Anomalies are localized (e.g., a person running affects only part of the frame). Mean error dilutes the signal by averaging with normal pixels.

### 2. Temporal Smoothing (+13%)
```python
from scipy.ndimage import gaussian_filter1d

def smooth_scores(scores, video_fnums, sigma=3):
    smoothed = {}
    for vid, fnums in video_fnums.items():
        vals = np.array([scores[f"{vid}_{fn}"] for fn in fnums])
        vals_smooth = gaussian_filter1d(vals, sigma=sigma)
        for i, fn in enumerate(fnums):
            smoothed[f"{vid}_{fn}"] = vals_smooth[i]
    return smoothed
```
**Why:** Anomalies span multiple consecutive frames. Smoothing propagates high scores to neighbors and reduces single-frame noise.

### 3. Multi-Scale Processing (+4.5%)
```python
# Train separate autoencoders at different resolutions
scales = {
    'small': (64, 64),    # Global patterns
    'medium': (128, 128), # Balanced detail
    'large': (256, 256),  # Fine details
}

# Fuse with MAX across scales
final_score = max(score_small, score_medium, score_large)
```
**Why:** Different anomalies are visible at different spatial scales. MAX fusion ensures if ANY scale detects an anomaly, it gets flagged.

### 4. No Skip Connections
```python
# DON'T use U-Net style skip connections for anomaly detection!
# They allow the model to reconstruct anomalies well, defeating the purpose.
```

---

## Final Architecture

### Multi-Scale Convolutional Autoencoder

```
Scale: 64x64 (Small)
├── Encoder: Conv2d(3→32→64→128→256→512) + BatchNorm + LeakyReLU
├── Bottleneck: 2x2 spatial, 64-dim latent
└── Decoder: ConvTranspose2d(512→256→128→64→32→3) + Tanh

Scale: 128x128 (Medium)
├── Encoder: Conv2d(3→32→64→128→256→512) + BatchNorm + LeakyReLU
├── Bottleneck: 4x4 spatial, 128-dim latent
└── Decoder: ConvTranspose2d(512→256→128→64→32→3) + Tanh

Scale: 256x256 (Large)
├── Encoder: Conv2d(3→32→64→128→256→512) + BatchNorm + LeakyReLU
├── Bottleneck: 8x8 spatial, 256-dim latent
└── Decoder: ConvTranspose2d(512→256→128→64→32→3) + Tanh
```

### Pipeline
```
Input Frames → Multi-Scale AE → MAX Error → Ensemble Average → 
Gaussian Smoothing (σ=3) → MAX Fusion → Final Anomaly Scores
```

---

## What Didn't Work

| Approach | Score | Why It Failed |
|----------|-------|---------------|
| U-Net | 0.33 | Skip connections let model reconstruct anomalies well |
| 3D CNN | 0.53 | Temporal convolutions didn't add discriminative power |
| ConvLSTM | 0.41-0.45 | Frame prediction is wrong task - anomalies aren't unpredictable |
| Optical Flow | 0.31 | Anomalies are appearance-based, not motion-based |
| Per-Video Normalization | 0.54 | Loses cross-video magnitude comparison |

---

## Installation and Usage

### Requirements
```bash
pip install -r requirements.txt
```

### Running the Final Solution
```python
# 1. Load and preprocess data at multiple scales
# 2. Train autoencoders for each scale
# 3. Compute MAX reconstruction error
# 4. Apply Gaussian smoothing (σ=3)
# 5. Fuse scores with MAX across scales

# See Phase5_Exp1_MultiScale.ipynb for complete implementation
```

---

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Loss Function | MSE |
| Optimizer | Adam |
| Learning Rate | 2e-4 |
| Epochs | 25 (with early stopping) |
| Patience | 5 |
| Batch Size | 64 (small), 32 (medium), 8 (large) |
| Ensemble | 2 models per scale |
| Smoothing | Gaussian σ=3 |

---

## Results

### Fusion Comparison
| Method | Score |
|--------|-------|
| MAX across scales | **0.6310** |
| Medium-heavy (60%) | 0.6281 |
| Large only | 0.6233 |
| Equal weights | 0.6226 |

---

## Lessons Learned

1. **Scoring method > Architecture complexity:** MAX vs Mean error provided larger improvement than any architectural change.

2. **Post-processing is powerful:** Temporal smoothing and multi-scale fusion provided major gains without changing the core model.

3. **Understand your data:** Anomalies in this dataset are appearance-based, not motion-based. This insight saved time by avoiding temporal approaches.

4. **Intentionally limit model capacity:** For anomaly detection, models should be "bad" at reconstructing anomalies.

5. **Capitalize on what works:** Rather than replacing the Simple ConvAE, we built on it with smoothing and multi-scale processing.
