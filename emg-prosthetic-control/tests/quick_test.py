import yaml
from src.data.loader import load_emg_imu
from src.features.windowing import window_and_extract_features
import numpy as np

# Load configuration
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

FS_EMG = config["features"]["fs_emg"]
FS_IMU = config["features"]["fs_imu"]
WINDOW_SEC = config["features"]["window_sec"]
OVERLAP_SEC = config["features"]["overlap_sec"]

# Try loading one file
print("Testing data loading...")
emg_data, imu_data, _, _, _ = load_emg_imu(
    "data/20251202/0.1_20251202.csv",
    fs_emg=FS_EMG,
    fs_imu=FS_IMU
)
print(f"✓ Loaded EMG: {len(emg_data)} sensors")
print(f"✓ Loaded IMU: {len(imu_data)} sensors")

# Try feature extraction
print("\nTesting feature extraction...")
X = window_and_extract_features(
    emg_data,
    imu_data,
    fs_emg=FS_EMG,
    fs_imu=FS_IMU,
    window_sec=WINDOW_SEC,
    overlap_sec=OVERLAP_SEC
)
print(f"✓ Extracted features: {X.shape}")

print("\n✓ All tests passed!")
