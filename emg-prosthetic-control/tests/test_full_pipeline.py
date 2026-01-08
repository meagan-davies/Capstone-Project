"""
Full Pipeline Test Script
-------------------------

This script tests the full EMG+IMU pipeline:
1. Loads a sample CSV data file
2. Extracts EMG and IMU features using windowing
3. Trains an LDA classifier
4. Evaluates the classifier
5. Saves and reloads the model package
"""

import os
import yaml
import numpy as np
from src.data.loader import load_emg_imu
from src.features.windowing import window_and_extract_features
from src.models.classifier import train_and_evaluate
from src.models.model_utils import save_model_package, load_model_package

# -------------------------------
# 1. Load configuration
# -------------------------------
CONFIG_PATH = "config/config.yaml"

with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

FS_EMG = config["features"]["fs_emg"]
FS_IMU = config["features"]["fs_imu"]

WINDOW_SEC = config["features"].get("window_sec", 0.2)
OVERLAP_SEC = config["features"].get("overlap_sec", 0.1)

# -------------------------------
# 2. Load a sample CSV file
# -------------------------------
SAMPLE_FILE = "data/20251202/0.1_20251202.csv"

print("\n=== Testing Data Loading ===")
emg_data, imu_data, _, fs_emg, fs_imu = load_emg_imu(
    SAMPLE_FILE,
    fs_emg=FS_EMG,
    fs_imu=FS_IMU
)
print(f"✓ Loaded EMG sensors: {len(emg_data)}")
print(f"✓ Loaded IMU sensors: {len(imu_data)}")

# -------------------------------
# 3. Extract features
# -------------------------------
print("\n=== Testing Feature Extraction ===")
X = window_and_extract_features(
    emg_data, imu_data, fs_emg, fs_imu,
    window_sec=WINDOW_SEC,
    overlap_sec=OVERLAP_SEC
)
print(f"✓ Extracted features: {X.shape}")

# For testing, generate synthetic labels
y = np.random.randint(0, 4, size=X.shape[0])

# -------------------------------
# 4. Train classifier
# -------------------------------
print("\n=== Testing Classifier Training ===")
clf, scaler, results = train_and_evaluate(
    X, y,
    test_size=0.2,
    scaler_type='robust',
    use_cv=True,
    cv_folds=3,
    class_names=['Neutral', 'Pinching', 'Grasping', 'Zipping'],
    verbose=True
)

# -------------------------------
# 5. Save model package
# -------------------------------
print("\n=== Testing Model Saving ===")
save_path = save_model_package(
    clf, scaler,
    model_name="test_pipeline_model",
    class_names={0:'Neutral', 1:'Pinching', 2:'Grasping', 3:'Zipping'},
    results=results,
    config=config
)
print(f"✓ Model saved at: {save_path}")

# -------------------------------
# 6. Load model package
# -------------------------------
print("\n=== Testing Model Loading ===")
package = load_model_package("test_pipeline_model")
clf_loaded = package['model']
scaler_loaded = package['scaler']
metadata_loaded = package['metadata']

print(f"✓ Loaded model type: {type(clf_loaded).__name__}")
print(f"✓ Loaded scaler type: {type(scaler_loaded).__name__}")
print(f"✓ Loaded model classes: {metadata_loaded.get('class_names', {})}")

# -------------------------------
# 7. Test prediction
# -------------------------------
print("\n=== Testing Prediction ===")
X_test_scaled = scaler_loaded.transform(X)
y_pred = clf_loaded.predict(X_test_scaled)
print(f"✓ Predictions shape: {y_pred.shape}")
print(f"✓ Unique predicted classes: {np.unique(y_pred)}")

print("\n=== FULL PIPELINE TEST PASSED ===")
