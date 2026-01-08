# test_quick.py
from src.data.loader import load_emg_imu
from src.features.emg_features import extract_emg_features
from src.features.windowing import window_and_extract_features
import numpy as np

# Try loading one file
print("Testing data loading...")
emg_data, imu_data, _, fs_emg, fs_imu = load_emg_imu("data/20251202/0.1_hannah_20251202.csv")
print(f"✓ Loaded EMG: {len(emg_data)} sensors")
print(f"✓ Loaded IMU: {len(imu_data)} sensors")

# Try feature extraction
print("\nTesting feature extraction...")
X = window_and_extract_features(emg_data, imu_data, fs_emg, fs_imu)
print(f"✓ Extracted features: {X.shape}")

print("\n✓ All tests passed!")