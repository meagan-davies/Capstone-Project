#!/usr/bin/env python
"""
Feature Count Diagnostic Test Script

This script analyzes your training data to determine:
1. How many sensors/channels were used
2. What feature extraction produces
3. Why you're getting 456 features

Place this in your tests/ folder and run:
    python tests/test_feature_count.py --csv path/to/your/training.csv
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.loader import load_emg_imu
from src.features.windowing import window_and_extract_features
from src.features.emg_features import extract_emg_features
from src.features.imu_features import extract_imu_features


def analyze_data_structure(csv_path, fs_emg=1259, fs_imu=148):
    """
    Analyze the structure of loaded training data.
    """
    print("="*70)
    print("TRAINING DATA STRUCTURE ANALYSIS")
    print("="*70)
    
    # Load data
    print(f"\nLoading data from: {csv_path}")
    emg_data, imu_data, time_data, fs_emg, fs_imu = load_emg_imu(
        csv_path, fs_emg, fs_imu
    )
    
    print(f"✓ Data loaded")
    print(f"  EMG sampling rate: {fs_emg} Hz")
    print(f"  IMU sampling rate: {fs_imu} Hz")
    
    # Analyze EMG structure
    print("\n" + "="*70)
    print("EMG DATA STRUCTURE")
    print("="*70)
    print(f"Number of EMG sensors: {len(emg_data)}")
    
    total_emg_channels = 0
    total_emg_samples = 0
    
    for sensor_name, data in sorted(emg_data.items()):
        n_samples = data.shape[0]
        n_channels = data.shape[1] if len(data.shape) > 1 else 1
        total_emg_channels += n_channels
        total_emg_samples = max(total_emg_samples, n_samples)
        
        print(f"  {sensor_name:40s}: shape = {str(data.shape):20s} ({n_samples:6d} samples, {n_channels} ch)")
    
    print(f"\nTotal EMG channels across all sensors: {total_emg_channels}")
    print(f"Max samples: {total_emg_samples}")
    
    # Analyze IMU structure
    print("\n" + "="*70)
    print("IMU DATA STRUCTURE")
    print("="*70)
    print(f"Number of IMU sensors: {len(imu_data)}")
    
    total_imu_channels = 0
    total_imu_samples = 0
    
    for sensor_name, data in sorted(imu_data.items()):
        n_samples = data.shape[0]
        n_channels = data.shape[1] if len(data.shape) > 1 else 1
        total_imu_channels += n_channels
        total_imu_samples = max(total_imu_samples, n_samples)
        
        print(f"  {sensor_name:40s}: shape = {str(data.shape):20s} ({n_samples:6d} samples, {n_channels} ch)")
    
    print(f"\nTotal IMU channels across all sensors: {total_imu_channels}")
    print(f"Max samples: {total_imu_samples}")
    
    return emg_data, imu_data, fs_emg, fs_imu


def test_feature_extraction(emg_data, imu_data, fs_emg, fs_imu, 
                            window_sec=0.2, overlap_sec=0.1):
    """
    Test feature extraction and count features.
    """
    print("\n" + "="*70)
    print("FEATURE EXTRACTION TEST")
    print("="*70)
    
    print(f"\nWindow parameters:")
    print(f"  Window duration: {window_sec} seconds")
    print(f"  Overlap: {overlap_sec} seconds")
    print(f"  EMG window size: {int(window_sec * fs_emg)} samples")
    print(f"  IMU window size: {int(window_sec * fs_imu)} samples")
    
    # Extract features
    print(f"\nExtracting features...")
    features = window_and_extract_features(
        emg_data, imu_data, fs_emg, fs_imu, window_sec, overlap_sec
    )
    
    print(f"✓ Features extracted")
    print(f"\nFeature matrix shape: {features.shape}")
    print(f"  Number of windows: {features.shape[0]}")
    print(f"  Features per window: {features.shape[1]}")
    
    return features


def analyze_feature_breakdown(emg_data, imu_data, fs_emg, fs_imu):
    """
    Manually calculate features for one window to understand the breakdown.
    """
    print("\n" + "="*70)
    print("DETAILED FEATURE BREAKDOWN (Single Window)")
    print("="*70)
    
    window_sec = 0.2
    emg_win_size = int(window_sec * fs_emg)
    imu_win_size = int(window_sec * fs_imu)
    
    emg_feature_count = 0
    imu_feature_count = 0
    
    # EMG features
    print("\nEMG Feature Extraction:")
    emg_sensors = sorted(emg_data.keys())
    for sensor in emg_sensors:
        data = emg_data[sensor]
        if len(data) < emg_win_size:
            continue
        
        window = data[:emg_win_size].flatten()
        feats = extract_emg_features(window, fs_emg)
        emg_feature_count += len(feats)
        
        print(f"  {sensor:40s}: {len(feats)} features")
    
    print(f"\n  Total EMG features: {emg_feature_count}")
    
    # IMU features
    print("\nIMU Feature Extraction:")
    imu_sensors = sorted(imu_data.keys())
    for sensor in imu_sensors:
        data = imu_data[sensor]
        if len(data) < imu_win_size:
            continue
        
        window = data[:imu_win_size]
        feats = extract_imu_features(window)
        imu_feature_count += len(feats)
        
        print(f"  {sensor:40s}: input shape = {window.shape}, features = {len(feats)}")
    
    print(f"\n  Total IMU features: {imu_feature_count}")
    
    # Total
    total = emg_feature_count + imu_feature_count
    print("\n" + "="*70)
    print(f"TOTAL FEATURES: {total}")
    print(f"Expected: 456")
    print(f"Difference: {456 - total}")
    
    if total == 456:
        print("\n✓✓✓ PERFECT MATCH! ✓✓✓")
    elif abs(total - 456) <= 5:
        print("\n⚠ Close match - might be rounding or edge case")
    else:
        print("\n✗ Significant mismatch - investigate further")
    
    print("="*70)
    
    return emg_feature_count, imu_feature_count, total


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training data feature count"
    )
    parser.add_argument(
        '--csv',
        type=str,
        required=True,
        help='Path to training CSV file'
    )
    parser.add_argument(
        '--fs-emg',
        type=float,
        default=1259,
        help='EMG sampling frequency (Hz)'
    )
    parser.add_argument(
        '--fs-imu',
        type=float,
        default=148,
        help='IMU sampling frequency (Hz)'
    )
    parser.add_argument(
        '--window',
        type=float,
        default=0.2,
        help='Window duration (seconds)'
    )
    parser.add_argument(
        '--overlap',
        type=float,
        default=0.1,
        help='Window overlap (seconds)'
    )
    
    args = parser.parse_args()
    
    # Check if file exists
    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"✗ Error: File not found: {csv_path}")
        print(f"\nCurrent directory: {Path.cwd()}")
        print(f"\nLooking for CSV files in data/:")
        data_dir = project_root / "data"
        if data_dir.exists():
            csv_files = list(data_dir.glob("**/*.csv"))
            if csv_files:
                print("\nFound CSV files:")
                for f in csv_files:
                    print(f"  - {f.relative_to(project_root)}")
            else:
                print("  No CSV files found")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print("FEATURE COUNT DIAGNOSTIC TEST")
    print("="*70)
    print(f"CSV file: {csv_path}")
    print(f"EMG sampling rate: {args.fs_emg} Hz")
    print(f"IMU sampling rate: {args.fs_imu} Hz")
    print(f"Window: {args.window}s, Overlap: {args.overlap}s")
    
    # Step 1: Analyze data structure
    emg_data, imu_data, fs_emg, fs_imu = analyze_data_structure(
        csv_path, args.fs_emg, args.fs_imu
    )
    
    # Step 2: Test feature extraction
    features = test_feature_extraction(
        emg_data, imu_data, fs_emg, fs_imu, 
        args.window, args.overlap
    )
    
    # Step 3: Detailed breakdown
    emg_feats, imu_feats, total_feats = analyze_feature_breakdown(
        emg_data, imu_data, fs_emg, fs_imu
    )
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"EMG sensors: {len(emg_data)}")
    print(f"IMU sensors: {len(imu_data)}")
    print(f"EMG features: {emg_feats}")
    print(f"IMU features: {imu_feats}")
    print(f"Total features: {total_feats}")
    print(f"Expected: 456")
    
    if total_feats != 456:
        print("\n" + "="*70)
        print("TROUBLESHOOTING")
        print("="*70)
        print("\nPossible reasons for mismatch:")
        print("1. This CSV has different sensors than the training data")
        print("2. The data_loader was modified between training and now")
        print("3. Different window parameters were used in training")
        print("4. Some sensors were excluded during training")
        print("\nNext steps:")
        print("- Find the EXACT CSV file used for training")
        print("- Check git history for changes to data_loader.py")
        print("- Verify window_sec and overlap_sec used in training")
    else:
        print("\n✓ Feature count matches! This is the correct data structure.")
    
    print("="*70)


if __name__ == "__main__":
    main()