#!/usr/bin/env python
"""
EMG+IMU Gesture Classification - Training Script
"""

import sys
import argparse
import os
import glob
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Data & Features
from src.data.data_loader import load_emg_imu
from src.features.windowing import window_and_extract_features

# Model system
from src.models.classifier import train_model
from src.models import model_utils


# Argument Parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EMG+IMU gesture classification model"
    )

    parser.add_argument('--data', type=str, nargs='+', required=True)
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--model-dir', type=str, default='models')
    parser.add_argument('--test-size', type=float, default=0.2)
    parser.add_argument('--cv-folds', type=int, default=5)
    parser.add_argument('--no-cv', action='store_true')
    parser.add_argument('--scaler', type=str, default='robust',
                        choices=['standard', 'robust', 'minmax'])
    parser.add_argument('--window', type=float, default=0.2)
    parser.add_argument('--overlap', type=float, default=0.1)
    parser.add_argument('--fs-emg', type=float, default=1259)
    parser.add_argument('--fs-imu', type=float, default=148)
    parser.add_argument('--verbose', action='store_true')

    return parser.parse_args()

# Data Loading
def get_trials_from_folder(data_dir):
    trial_files = glob.glob(os.path.join(data_dir, "*.csv"))
    class_trials = {}

    for f in trial_files:
        basename = os.path.basename(f)
        try:
            class_label = int(basename.split(".")[0])
            class_trials.setdefault(class_label, []).append(f)
        except (ValueError, IndexError):
            print(f"⚠ Skipping unexpected file: {basename}")

    return class_trials


def get_trials_from_multiple_folders(data_dirs):
    if isinstance(data_dirs, str):
        data_dirs = [data_dirs]

    combined_trials = {}

    print(f"Scanning {len(data_dirs)} folder(s)...")

    for folder in data_dirs:
        if not os.path.exists(folder):
            print(f"⚠ Folder not found: {folder}")
            continue

        trials = get_trials_from_folder(folder)

        for label, files in trials.items():
            combined_trials.setdefault(label, []).extend(files)

    return combined_trials


def process_trial(csv_path, label, fs_emg, fs_imu, window_sec, overlap_sec):
    emg_dict, imu_dict, _, fs_emg_used, fs_imu_used = load_emg_imu(
        csv_path, fs_emg, fs_imu
    )

    X = window_and_extract_features(
        emg_dict,
        imu_dict,
        fs_emg_used,
        fs_imu_used,
        window_sec,
        overlap_sec,
        include_partial=True
    )

    y = np.full(X.shape[0], label)
    return X, y


def load_all_trials(class_trials, fs_emg, fs_imu, window_sec, overlap_sec):
    X_all = []
    y_all = []

    for label, files in sorted(class_trials.items()):
        print(f"\nProcessing Class {label} ({len(files)} files)...")

        for file in files:
            try:
                X_trial, y_trial = process_trial(
                    file, label, fs_emg, fs_imu, window_sec, overlap_sec
                )
                X_all.append(X_trial)
                y_all.append(y_trial)
            except Exception as e:
                print(f"✗ ERROR in {file}: {e}")

    if not X_all:
        raise ValueError("No data was successfully loaded.")

    return np.vstack(X_all), np.concatenate(y_all)

# Main
def main():
    args = parse_args()

    print("=" * 70)
    print("EMG + IMU CLASSIFICATION - TRAINING")
    print("=" * 70)

    # 1️⃣ Load data
    class_trials = get_trials_from_multiple_folders(args.data)

    if not class_trials:
        print("✗ No valid CSV files found.")
        sys.exit(1)

    X, y = load_all_trials(
        class_trials,
        args.fs_emg,
        args.fs_imu,
        args.window,
        args.overlap
    )

    print(f"\nTotal samples: {X.shape[0]}")
    print(f"Features per sample: {X.shape[1]}")

    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} samples")

    # 2️⃣ Train
    bundle, metrics, config = train_model(
        X,
        y,
        scaler_type=args.scaler,
        test_size=args.test_size,
        cv_folds=0 if args.no_cv else args.cv_folds,
        class_names=[f"Class {int(c)}" for c in unique],
        verbose=True,
    )

    # 3️⃣ Save
    save_path = model_utils.save_model_bundle(
        bundle=bundle,
        metrics=metrics,
        model_name=args.model_name,
        model_dir=args.model_dir,
    )

    print(f"\nTraining complete.")
    print(f"Model saved to: {save_path}")
    print(f"Test Accuracy: {metrics['test_accuracy']:.2%}")
    print(f"Macro F1: {metrics['f1_macro']:.4f}")

if __name__ == "__main__":
    main()