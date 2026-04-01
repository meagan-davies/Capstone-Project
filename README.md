# Real-Time Prosthetic Control and Embodiment Quantification System

## Overview

This capstone project implements two integrated machine learning systems:

1. **Prosthetic Control Model**  
   Real-time gesture classification using EMG and IMU signals from Delsys Trigno sensors.

2. **Embodiment Quantification Model**  
   A multimodal regression model that estimates user embodiment in real time (0–100 scale) using:
   - Leap Motion kinematics
   - Apple Watch physiological data
   - BioRadio biosignals

The project is structured as a modular monorepo with shared sensor infrastructure and separate ML pipelines for classification and regression.

---

## System Architecture
```
EMG + IMU → Gesture Classifier → Prosthetic Control Output

Leap Motion → Control Accuracy Metrics
BioRadio + Apple Watch + Leap → Embodiment Regression → Real-Time Score (0–100)
```

---

## Repository Structure
```
Capstone-Project/
├── docs/                    # Main feature usage explained
├── shared/                  # Sensor interfaces & utilities
├── prosthetic_control/      # Gesture classification system
├── embodiment_model/        # Embodiment regression system
├── integration/             # Combined system execution
├── data/                    # Raw and processed data (gitignored)
└── artifacts/               # Saved models (gitignored)
```

---

## Installation

### 1. Clone repository
```bash
git clone <repository-url>
cd Capstone-Project
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## Running the Systems

### Train Prosthetic Control Model
```bash
cd prosthetic_control
python scripts/train_model.py --data-dir ../data/raw/prosthetic_control
```

### Run Real-Time Prosthetic Classification
```bash
cd prosthetic_control
python scripts/realtime_classify.py --model-name my_control_model
```

### Train Embodiment Model
```bash
cd embodiment_model
python scripts/train_model.py --data-dir ../data/raw/embodiment
```

### Evaluate Embodiment Model
```bash
cd embodiment_model
python scripts/evaluate_model.py --model-name my_embodiment_model
```

### Run Integrated Real-Time System
```bash
python integration/combined_pipeline.py
```

---

## Embodiment Score

The embodiment model outputs a continuous score between 0–100:

- **0–25**: Low embodiment
- **26–50**: Moderate embodiment  
- **51–75**: Good embodiment
- **76–100**: High embodiment

The score is computed from:
- Control accuracy metrics (tracking error, smoothness, latency)
- Physiological signals (HRV, EMG activation, EDA arousal)
- Cross-sensor synchrony

---

## Data Collection

### Prosthetic Control
- **Sensors**: Delsys Trigno (8 EMG channels + IMU)
- **Gestures**: Neutral, Pinching, Grasping, Zipping

### Embodiment
- **Sensors**: BioRadio (EDA), Apple Watch, Leap Motion
- **Conditions**: Varying control accuracy (baseline, lag, jitter)

Raw and processed data are excluded from version control. To reproduce experiments, place datasets inside:
```
data/raw/prosthetic_control/
data/raw/embodiment/
```

---

## Key Features

### Prosthetic Control
- ✓ Real-time gesture classification (<100ms latency)
- ✓ Multi-class classification (4 gestures)
- ✓ 231 features per window (EMG + IMU)
- ✓ Cross fold validation

### Embodiment Model
- ✓ Multi-sensor fusion (3 modalities)
- ✓ Automatic feature selection (~30 features from 150+ candidates)
- ✓ Regression with interpretable formula extraction
- ✓ Real-time embodiment monitoring

### Integration
- ✓ Cross-model validation
- ✓ Control accuracy vs embodiment correlation analysis
- ✓ Combined real-time pipeline

---

## Performance Targets

| System | Metric | Target |
|--------|--------|--------|
| Prosthetic Control | Accuracy | >85% |
| Prosthetic Control | Latency | <100ms |
| Embodiment Model | R² (LOSO-CV) | >0.50 |
| Embodiment Model | MAE | <12 points |

---

## Documentation

- [Prosthetic Control README](./prosthetic_control/README.md)
- [Embodiment Model README](./embodiment_model/README.md)
- [Installation Guide](./docs/installation.md)

---

### Project structure follows:
- Shared utilities in `shared/`
- Project-specific code in module directories
- Scripts for training/evaluation in `scripts/`
- Saved models in `artifacts/` (gitignored)

---

## Author

**Meagan Davies**  
Combined Degree in Astrophysics & Biomedical Engineering  
University of Calgary

---

## Notes

- Real-time performance target: <100 ms latency (prosthetic control)
- Models saved in `artifacts/` (excluded from version control)
- Leave-One-Subject-Out cross-validation ensures generalization
- Feature importance analysis available in notebooks

## Acknowledgments

- Delsys Inc. for Trigno API access
- University of Calgary Biomedical Engineering department
- HERO Lab, capstone supervisor Dr. Junho Park and advisors