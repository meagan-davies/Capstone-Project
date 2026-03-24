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

from src.data.data_loader import load_emg_imu
from src.features.windowing import window_and_extract_features
from src.models.classifier import train_model
from src.models import model_utils


# ── Argument Parsing ──────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train EMG+IMU gesture classification model"
    )
    parser.add_argument('--data',       type=str, nargs='+', required=True)
    parser.add_argument('--model-name', type=str, default=None)
    parser.add_argument('--model-dir',  type=str, default='models')
    parser.add_argument('--test-size',  type=float, default=0.2)
    parser.add_argument('--cv-folds',   type=int,   default=5)
    parser.add_argument('--no-cv',      action='store_true')
    parser.add_argument('--classifier', type=str, default='lda', 
                        choices=['lda', 'svm'], help='Classifier type: lda or svm')
    parser.add_argument('--scaler',     type=str,   default='robust',
                        choices=['standard', 'robust', 'minmax'])
    parser.add_argument('--window',     type=float, default=0.2)
    parser.add_argument('--overlap',    type=float, default=0.1)
    # fs-emg / fs-imu are now fallback values only.
    # Actual per-sensor rates are read directly from the CSV header.
    parser.add_argument('--fs-emg',     type=float, default=962.963,
                        help='Fallback EMG fs if not found in CSV header')
    parser.add_argument('--fs-imu',     type=float, default=148.1481)
    parser.add_argument('--verbose',    action='store_true')
    return parser.parse_args()


# ── Data Loading ──────────────────────────────────────────────────────────────

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
        for label, files in get_trials_from_folder(folder).items():
            combined_trials.setdefault(label, []).extend(files)
    return combined_trials


def process_trial(csv_path, label, fs_emg_fallback, fs_imu, window_sec, overlap_sec):
    """
    Load one CSV trial and extract windowed features.
    emg_fs_map is read from the CSV header by load_emg_imu.
    """
    emg_dict, imu_dict, _, emg_fs_map, fs_imu_used = load_emg_imu(
        csv_path,
        fs_emg=fs_emg_fallback,
        fs_imu=fs_imu
    )

    X = window_and_extract_features(
        emg_dict,
        imu_dict,
        emg_fs_map,       # per-sensor fs — Avanti=962.963, Galileo=1259.2593
        fs_imu_used,
        window_sec,
        overlap_sec,
        include_partial=True
    )

    y = np.full(X.shape[0], label)
    return X, y


def load_all_trials(class_trials, fs_emg_fallback, fs_imu, window_sec, overlap_sec):
    X_all, y_all = [], []
    for label, files in sorted(class_trials.items()):
        print(f"\nProcessing Class {label} ({len(files)} files)...")
        for file in files:
            try:
                X_trial, y_trial = process_trial(
                    file, label, fs_emg_fallback, fs_imu, window_sec, overlap_sec
                )
                X_all.append(X_trial)
                y_all.append(y_trial)
                print(f"  ✓ {os.path.basename(file)} → {X_trial.shape[0]} windows, "
                      f"{X_trial.shape[1]} features")
            except Exception as e:
                print(f"  ✗ ERROR in {file}: {e}")

    if not X_all:
        raise ValueError("No data was successfully loaded.")

    return np.vstack(X_all), np.concatenate(y_all)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    print("=" * 70)
    print("EMG + IMU CLASSIFICATION - TRAINING")
    print("=" * 70)
    print(f"  Fallback EMG fs : {args.fs_emg} Hz  (overridden per-sensor by CSV header)")
    print(f"  IMU fs          : {args.fs_imu} Hz")
    print(f"  Window          : {args.window}s  |  Overlap: {args.overlap}s")

    # 1. Discover trial files
    class_trials = get_trials_from_multiple_folders(args.data)
    if not class_trials:
        print("✗ No valid CSV files found.")
        sys.exit(1)

    # 2. Load & window all trials
    X, y = load_all_trials(
        class_trials,
        fs_emg_fallback=args.fs_emg,
        fs_imu=args.fs_imu,
        window_sec=args.window,
        overlap_sec=args.overlap
    )

    print(f"\nTotal windows  : {X.shape[0]}")
    print(f"Features/window: {X.shape[1]}")
    unique, counts = np.unique(y, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"  Class {u}: {c} windows")

    # 3. Train
    bundle, metrics, config = train_model(
        X,
        y,
        scaler_type=args.scaler,
        classifier_type=args.classifier,
        test_size=args.test_size,
        cv_folds=0 if args.no_cv else args.cv_folds,
        class_names=[f"Class {int(c)}" for c in unique],
        verbose=True,
    )

    # 4. Save
    save_path = model_utils.save_model_bundle(
        bundle=bundle,
        metrics=metrics,
        model_name=args.model_name,
        model_dir=args.model_dir,
    )

    print(f"\nTraining complete.")
    print(f"Model saved to : {save_path}")
    print(f"Test Accuracy  : {metrics['test_accuracy']:.2%}")
    print(f"Macro F1       : {metrics['f1_macro']:.4f}")


if __name__ == "__main__":
    main()