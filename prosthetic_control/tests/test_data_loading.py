# test_data_pipeline.py

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.train_model import (
    get_trials_from_multiple_folders,
    load_all_trials,
    process_trial
)

# ---------------- CONFIG ----------------
DATA_DIR = "data/20260113"
WINDOW_SEC = 0.2
OVERLAP_SEC = 0.1
FS_EMG = 1259
FS_IMU = 148
VERBOSE = True

# ---------------- TEST SCRIPT ----------------
def test_data_pipeline():
    print("\n=== TESTING DATA LOADING & FEATURE EXTRACTION ===\n")

    # Step 1: scan folder
    class_trials, folder_info = get_trials_from_multiple_folders([DATA_DIR], verbose=VERBOSE)
    print("\nFolder scan complete.\n")
    
    # Step 2: inspect files and sensors
    for label, files in class_trials.items():
        print(f"\nClass {label}: {len(files)} files")
        for file in files:
            print(f"  Processing {os.path.basename(file)}")
            try:
                # Use existing process_trial to load and window features
                X_trial, y_trial = process_trial(
                    file, label, FS_EMG, FS_IMU, WINDOW_SEC, OVERLAP_SEC, verbose=VERBOSE
                )
                print(f"    → Features: {X_trial.shape}, Labels: {y_trial.shape}")
                
                # Check for NaNs
                nan_count = np.isnan(X_trial).sum()
                if nan_count > 0:
                    print(f"    ⚠ Warning: {nan_count} NaNs detected in features")
            except Exception as e:
                print(f"    ✗ ERROR: {e}")
                continue

    # Step 3: load all trials together
    print("\nLoading all trials into combined dataset...")
    try:
        X_all, y_all = load_all_trials(
            class_trials, FS_EMG, FS_IMU, WINDOW_SEC, OVERLAP_SEC, verbose=VERBOSE
        )
        print(f"\nCombined dataset: {X_all.shape[0]} samples, {X_all.shape[1]} features")
        print(f"Labels distribution:")
        for cls in np.unique(y_all):
            print(f"  Class {cls}: {(y_all == cls).sum()} samples")
        
        nan_count = np.isnan(X_all).sum()
        print(f"Total NaNs in combined dataset: {nan_count}")

    except Exception as e:
        print(f"✗ ERROR loading all trials: {e}")


if __name__ == "__main__":
    test_data_pipeline()
