# Pixel_Play_2026
This repository documents my approach to video anomaly detection on pedestrian surveillance videos.

## Problem
Detect anomalous events (e.g., running, throwing objects, unusual motion) in videos where only normal data is available during training.

## Approach
- Careful data exploration and preprocessing
- Learn normal patterns from training videos
- Detect anomalies as deviations using reconstruction and motion-based signals
- Focus on ranking-based evaluation metrics (AUC)

## Structure
- `notebooks/` – experiments and analysis
- `notes/` – reasoning, observations, and failed attempts
- `submissions/` – submission-related information

## Status
- Day 1: Data exploration and preprocessing (in progress)
