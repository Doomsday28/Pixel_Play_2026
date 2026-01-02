# Pixel Play 2026 – ConvAutoencoder Baseline (Post-Mortem)

This repository documents an initial baseline approach for the **Pixel Play 2026 video anomaly detection challenge**, along with a detailed explanation of **why the approach failed** and what was learned from it.

---

## 1. Problem Overview

- **Task**: Frame-level video anomaly detection  
- **Dataset**: Avenue-style surveillance videos  
- **Output**: One anomaly score per frame  
- **Metric**: **AUC (Area Under the ROC Curve)**  

⚠️ **Important constraint**:  
Every frame in the test set **must have exactly one prediction**. Missing even a single frame causes submission errors or severe score degradation.

---

## 2. Initial Approach: ConvAutoencoder

### Why ConvAutoencoder?
Convolutional Autoencoders (CAE) are a common unsupervised baseline for anomaly detection:

> Train only on normal data → learn normal patterns → anomalies produce higher reconstruction error.

This approach is widely used in early baselines for datasets like UCSD Ped and Avenue.

---

## 3. Frame Difference Modeling

Instead of raw frames, we trained the model on **frame differences**:

\[
\text{diff}_t = |I_t - I_{t-1}|
\]

### Motivation
- Suppress static background
- Emphasize motion
- Reduce appearance variance
- Simplify the learning task

Each training sample is a **single-channel grayscale difference image**.

---

## 4. Model Architecture

### ConvAutoencoder
- **Encoder**: Stacked Conv2D + ReLU layers with downsampling
- **Decoder**: ConvTranspose2D layers with upsampling
- **Final activation**: Sigmoid
- **Loss**: Mean Squared Error (MSE)

Training was stable and converged smoothly.

---

## 5. Anomaly Scoring

### Initial Scoring
- Pixel-wise reconstruction error

### Improved Scoring
- **Patch-based reconstruction error**
- Error map pooled into patches
- Top-K highest-error patches averaged

This was done to better capture **localized anomalies**.

---

## 6. Critical Issue #1: Missing First Frame

### What Happened?
Frame differences are undefined for the first frame of each video:

\[
\text{diff}_0 = |I_0 - I_{-1}| \quad \text{(undefined)}
\]

As a result:
- The first frame of **every video was skipped**
- Total predictions became **11685 instead of 11706**
- Kaggle rejected or penalized submissions

---

## 7. Attempted Fix: Manual Frame Completion

To satisfy Kaggle’s submission format:
- All missing frame IDs were reconstructed
- Missing frames were assigned a default score (e.g., `0.0`)
- Rows were appended, sorted, and deduplicated

This produced:
- Correct number of rows
- No missing IDs
- Valid submission CSV

---

## 8. Why This Fix Failed

Although the submission became valid, **the model has no information for the first frame**.

Assigning a score to frame 0 is:
- Arbitrary
- Uninformed
- Pure noise

### Why this hurts AUC
- AUC depends on **relative ranking**
- Injecting meaningless scores distorts rankings
- Repeating this across many videos compounds the error

Result:
- **AUC dropped significantly** after fixing missing frames

---

## 9. Fundamental Model Failure

### Faulty Assumption
> “Anomalies produce higher frame-difference reconstruction error.”

### Reality of Avenue / Pixel Play
- Normal videos contain:
  - Camera jitter
  - Illumination changes
  - Shadows
  - Background motion
- Abnormal events:
  - Often temporally smooth
  - Structured motion
  - Not necessarily high pixel differences

### What the Model Learns
- Background noise = normal
- Many anomalies = reconstructable
- Reconstruction error distributions overlap heavily

➡️ **Poor anomaly separation**  
➡️ **Low AUC (~0.20–0.35)**  

This is a **modeling limitation**, not an implementation bug.

---

## 10. Summary of Results

### What Worked
- Clean data pipeline
- Stable training
- Patch-based anomaly scoring
- Correct submission formatting
- No NaNs
- No missing frame IDs

### What Failed
- Core modeling assumption
- Frame-difference autoencoding is too weak
- No temporal modeling
- First-frame problem is fundamentally unfixable in this setup

---

## 11. Key Takeaway

This project implements a **valid but weak baseline**.

> ❌ ConvAutoencoder + frame difference is **not sufficient** for Pixel Play 2026.

Improving performance requires **changing the modeling paradigm**, not tuning hyperparameters.

---

## 12. Future Directions

Promising alternatives:
- Future frame prediction models
- Optical-flow-based anomaly scoring
- ConvLSTM or 3D CNNs
- Explicit temporal consistency modeling

---

## Final Note

This failure is **expected and informative**.  
Understanding *why* a baseline fails is a critical step toward building stronger anomaly detection models.
