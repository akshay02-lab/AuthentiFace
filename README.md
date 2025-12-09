# AuthentiFace 

AuthentiFace is a real-time authentication system that uses micro-expression analysis and liveness detection to secure face-based logins. It introduces a dual-head Vision Transformer capable of predicting both emotional state and live/spoof status from the same shared backbone.
## Dataset Download 

1. RAF-DB – Real-world Affective Faces Dataset

https://www.kaggle.com/datasets/ashishpatel26/raf-db

2. SAMM v1 – Micro-Expression Dataset

https://www.kaggle.com/datasets/sajidshahriar/samm-v1-micro-expression-dataset

3. CelebA-Spoof – Face Anti-Spoofing Dataset

https://www.kaggle.com/datasets/kpvisionlab/celeb-a-spoof-dataset

## Problem Summary
Traditional face authentication is vulnerable to:
- High-quality printed photos
- Screen replay attacks
- Deepfake videos

Common liveness cues (blinks, texture patterns, simple CNN-based PAD methods) are now easily spoofed. A stronger, involuntary biometric cue is required for modern security systems.

## Key Insight and Novelty
- Micro-expressions occur for less than 1/25th of a second and cannot be voluntarily controlled or reproduced by screens or deepfake systems.
- AuthentiFace is the first system to integrate micro-expression learning into liveness detection.
- A dual-head Vision Transformer jointly handles emotion classification and liveness prediction.
- Three-stage curriculum training improves generalization:
  - RAF-DB for macro-expression learning
  - SAMM for micro-expression fine-tuning
  - CelebA-Spoof for liveness classification

## System Overview
1. User accesses the login interface.
2. Webcam frames are captured and processed.
3. Each frame is passed through the ViT model to generate:
   - Emotion prediction
   - Live/spoof probability
4. Temporal smoothing ensures stable decisions across frames.
5. Decision logic:
   - Live → Login granted
   - Spoof → Access denied + automated email alert

### Features
- Fully edge-based inference
- Real-time visualization of liveness and emotion
- Stable prediction through smoothing
- Security alerts for confirmed spoof attempts

## Model Summary

### Input Pipeline
- Real-time webcam frames
- BGR → RGB conversion
- Center crop
- Resize to 224×224
- ImageNet-normalized tensor as model input

### ViT-Tiny Backbone
- Patch embedding (16×16 patches)
- Transformer encoder layers
- Global pooled embedding feeding both prediction heads

### Dual-Head Architecture
- Emotion Head: 7-class macro/micro expression classification
- Liveness Head: binary live/spoof classification
- Shared backbone improves feature reuse and micro-expression sensitivity

### Temporal Smoothing
- Rolling window of recent probabilities
- Moving average to reduce noise
- Threshold-based decision logic for reliable authorization

## Training Summary
### Stage 1 – RAF-DB
Learns general facial expression structure.

### Stage 2 – SAMM
Fine-tunes backbone for micro-expression sensitivity.

### Stage 3 – CelebA-Spoof
Trains the liveness head for presentation attack detection.

## Evaluation Summary

### Liveness Detection (CelebA-Spoof)
- Accuracy: 93%
- Precision (Spoof): 1.00
- No spoof misclassified as live
- Strong security and low false positives

### Emotion Classification (RAF-DB)
- Accuracy: 80%
- Macro F1: 0.7985
- Reliable across emotion categories
  

## System Challenges
- Integrating datasets with different structures
- Achieving stable real-time performance
- Handling noise and lighting variability
- Designing secure decision thresholds
- Creating an interpretable and responsive UI

## Limitations
- SAMM dataset size limits micro-expression diversity
- Sensitive to lighting and low-resolution cameras
- No depth/IR/rPPG sensing; relies only on RGB
- Possible demographic variation in micro-expression patterns

## Real-World Testing Summary
- Tested under varying lighting, occlusions, pose changes, and motion blur
- Temporal smoothing improved reliability
- User study (six participants) showed:
  - High usability
  - Successful detection of printed and replay attacks
  - Automated email alerts working as intended
