# Embodiment Quantification Model

Multi-sensor regression model for quantifying sense of embodiment during prosthetic control tasks.

## Overview

This model predicts embodiment scores (0-100) from multimodal sensor data:
- **Leap Motion**: Hand tracking and control accuracy
- **BioRadio**: EDA biosignals
- **Apple Watch**: Heart rate, HRV, motion

## Features

- ✓ Multi-sensor fusion (3 modalities)
- ✓ Automatic feature selection
- ✓ Regression with interpretable formula
- ✓ Real-time embodiment monitoring
- ✓ Leave-One-Subject-Out validation

## Installation

See main [README.md](../README.md) for installation instructions.

## Quick Start

### Train Model
```bash
python embodiment_model/scripts/train_model.py \
    --data-dir ../data/raw/embodiment \
    --model-type ridge \
    --output ../artifacts/embodiment_model
```

### Evaluate Model
```bash
python embodiment_model/scripts/evaluate_model.py \
    --model-path ../artifacts/embodiment_model/model.pkl \
    --test-data ../data/raw/embodiment/test
```

## Model Architecture

### Input Features (~30 selected from 150+ candidates)
1. **Control Accuracy** (Leap Motion)
   - Tracking error
   - Movement smoothness
   - Path efficiency
   - Response latency

2. **Physiological** (BioRadio + Apple Watch)
   - Heart rate variability (HRV)
   - EDA arousal levels

### Model Types
- **Ridge Regression** (default): Interpretable, fast
- **Random Forest**: Non-linear, robust
- **XGBoost**: Best performance

### Validation
- Leave-One-Subject-Out cross-validation
- Target R² > 0.50
- Target MAE < 12 points

## Configuration

Edit `config/model_config.yaml`:
```yaml
sensors:
  leap_motion:
    sampling_rate: 115
  bioradio:
    sampling_rate: 1000
    channels:
      emg: [0, 1, 2, 3]
      ecg: 4
      eda: 5
  apple_watch:
    sampling_rate: 50

model:
  type: ridge  # ridge, random_forest, xgboost
  feature_selection:
    method: lasso
    n_features: 30
  
validation:
  method: leave_one_subject_out
```

## Data Format

### Input
```
data/raw/embodiment/
└── YYYYMMDD/
    └── P01/
        └── P01_pre_trial001/            # Ground Truth Before
            ├── P01_1_bioradio
            ├── P01_1_leapmotion
            ├── P01_1_watch
            ├── P01_2_bioradio
            ├── P01_2_leapmotion
            ├── P01_2_watch
            ├── P01_3_bioradio
            ├── P01_3_leapmotion
            ├── P01_3_watch
            └── trial_001_labels.json
        └── P01_pros_trial001/            # Prosthetic
            ├── P01_1_bioradio
            ├── P01_1_leapmotion
            ├── P01_1_watch
            ├── P01_2_bioradio
            ├── P01_2_leapmotion
            ├── P01_2_watch
            ├── P01_3_bioradio
            ├── P01_3_leapmotion
            ├── P01_3_watch
            └── trial_001_labels.json
        └── P01_post_trial001/            # Ground Truth After
            ├── P01_1_bioradio
            ├── P01_1_leapmotion
            ├── P01_1_watch
            ├── P01_2_bioradio
            ├── P01_2_leapmotion
            ├── P01_2_watch
            ├── P01_3_bioradio
            ├── P01_3_leapmotion
            ├── P01_3_watch
            └── trial_001_labels.json
```

### Labels (trial_001_labels.json)
```json
{
  "participant_id":   "P01",
  "condition":        "pre",
  "trial_number":     1,
  "embodiment_score": 100.0,
  "session_start":    "2026-01-27T20:33:16+00:00",
  "session_end":      "2026-01-27T20:33:25+00:00"
}
```

## Performance

Expected metrics (LOSO-CV):
- R²: 0.50-0.70
- MAE: 8-12 points
- RMSE: 10-15 points