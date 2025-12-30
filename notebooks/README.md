##  Day 2: PCA-Based Video Anomaly Detection (Baseline)

### Objective

To build a **strong classical baseline** using **PCA reconstruction error**, and to understand:
- How anomalies differ from normal patterns
- The effect of normalization and score calibration
- Why temporal modeling is crucial in video anomaly detection

---

### Methodology

#### 1. Data Preparation
- Videos split into individual frames
- Frames resized to **64 × 64**
- Converted to **grayscale**
- Flattened into vectors

#### 2. Model
- `StandardScaler` for feature normalization
- `PCA (n_components = 100)`
- Frame reconstruction using `inverse_transform`
- **Anomaly score** = Mean Squared Reconstruction Error (MSE)

#### 3. Training Assumption
- Training data contains **only normal behavior**
- PCA learns a low-dimensional subspace representing normality

---

### 📊 Results

- Initial PCA baseline achieved **~0.35 AUC**
- This is a reasonable score for a **single-frame, appearance-only** method
- Provided a solid reference point for further improvements

---

### Issues Encountered

#### 1. Score Saturation due to Per-Video Normalization
- Min–max normalization performed **independently per video**
- Many frames collapsed to anomaly scores close to **1.0**
- Destroyed global ranking required for good AUC performance

#### 2. Over-Smoothing of Anomaly Peaks
- Gaussian temporal smoothing reduced noise
- But also suppressed sharp anomaly peaks
- Resulted in a **drop in AUC (~0.26)**

---

### Key Learnings

- AUC depends on **relative ranking**, not absolute score values
- Per-video normalization introduces bias
- Temporal smoothing must preserve anomaly peaks
- PCA alone cannot capture **motion or temporal dynamics**

---

### Conclusion (Day 2)

PCA-based reconstruction is a **useful conceptual baseline**, but:
- Lacks motion awareness
- Ignores temporal dependencies
- Is insufficient for high-quality video anomaly detection

This motivates the need for:
- Motion-aware features
- Temporal modeling
- Deep spatio-temporal architectures

---

## Next Steps

- **Day 3:** Motion-aware PCA using frame differencing
- **Day 4:** Deep temporal modeling with ConvLSTM Autoencoders

---

## 📂 Repository Structure

