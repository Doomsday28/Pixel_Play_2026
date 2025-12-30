# Pixel Play 2026 – Video Anomaly Detection

## Day 2: PCA-based Baseline (Unsupervised)

### Problem
Detect anomalous events in pedestrian surveillance videos
without frame-level labels.

---

### Approach Overview

We use an **unsupervised reconstruction-based anomaly detection**
approach using **Principal Component Analysis (PCA)**.

Key idea:
> PCA is trained on mostly-normal frames.
> Normal frames reconstruct well.
> Anomalous frames reconstruct poorly → higher reconstruction error.

---

### Pipeline

1. Extract frames from training videos (assumed normal)
2. Preprocess:
   - Resize to 64×64
   - Convert to grayscale
   - Flatten to vectors
3. Train PCA (100 components) on training frames
4. Compute reconstruction error per frame
5. Normalize errors per video to [0, 1]
6. Use normalized reconstruction error as anomaly score
7. Generate submission CSV for test videos

---

### Initial Issue Encountered

**Problem:**
The model initially predicted values close to `1.0` for almost all frames.

**Cause:**
Improper normalization of reconstruction errors using
global statistics instead of per-video statistics.

This caused score saturation and loss of ranking information.

---

### Resolution

- Errors were normalized **per video** using min–max scaling:
  
  ```text
  score = (error - min_error) / (max_error - min_error)

