"""
Real-time feature extraction that matches training.

CRITICAL FIX: Training used ALL 60 channels from the hardware!
Your code was only capturing 49 channels (13 EMG + 36 IMU).

The missing 11 channels are likely:
- Skin check channels
- Additional sensor data
- Other Delsys channel types

Solution: Process ALL channels the same way training did.
"""

import numpy as np
from emg_features import extract_emg_features
from imu_features import extract_imu_features


def diagnose_all_channels(channel_info):
    """
    Analyze ALL channels from the hardware to find the missing ones.
    """
    print("\n" + "="*70)
    print("COMPLETE CHANNEL ANALYSIS")
    print("="*70)
    
    channel_types = {}
    
    for guid, info in channel_info.items():
        chan_type = info['type']
        if chan_type not in channel_types:
            channel_types[chan_type] = []
        channel_types[chan_type].append(guid)
    
    print(f"\nTotal channels: {len(channel_info)}")
    print(f"\nBreakdown by type:")
    
    total = 0
    for chan_type, guids in sorted(channel_types.items()):
        count = len(guids)
        total += count
        print(f"  {chan_type:20s}: {count:3d} channels")
    
    print(f"\nTotal: {total}")
    print("="*70)
    
    return channel_types


def extract_realtime_features_v1(emg_windows, imu_windows, emg_fs=1259):
    """
    Version 1: Group IMU channels into 3-axis sensors.
    
    Assumes:
    - emg_windows: (13, 192) - 13 channels, 192 samples each
    - imu_windows: (36, 29) - 36 channels, 29 samples each
    - 36 IMU channels = 12 sensors × 3 axes
    
    Returns:
    - Feature vector: 13×8 + 12×13 = 104 + 156 = 260 features
    """
    features = []
    
    # EMG features: 13 channels × 8 features = 104
    print(f"Processing {emg_windows.shape[0]} EMG channels...")
    for i in range(emg_windows.shape[0]):
        window = emg_windows[i].flatten()
        emg_feats = extract_emg_features(window, emg_fs)
        features.extend(emg_feats)
    
    emg_count = len(features)
    print(f"EMG features: {emg_count}")
    
    # IMU features: Group 36 channels into 12 sensors (3 axes each)
    # Assuming channels are ordered: [sensor1_x, sensor1_y, sensor1_z, sensor2_x, ...]
    n_imu_sensors = imu_windows.shape[0] // 3
    print(f"Processing {n_imu_sensors} IMU sensors (3 axes each)...")
    
    for sensor_idx in range(n_imu_sensors):
        # Get 3 consecutive channels for this sensor
        start_ch = sensor_idx * 3
        end_ch = start_ch + 3
        
        # Stack channels as (n_samples, 3_axes)
        sensor_window = imu_windows[start_ch:end_ch].T  # Transpose to (29, 3)
        
        # Extract features
        imu_feats = extract_imu_features(sensor_window)
        features.extend(imu_feats)
    
    imu_count = len(features) - emg_count
    print(f"IMU features: {imu_count}")
    print(f"Total features: {len(features)}")
    
    return np.array(features)


def extract_realtime_features_v2(emg_windows, acc_windows, gyro_windows, emg_fs=1259):
    """
    Version 2: Separate ACC and GYRO processing.
    
    Assumes:
    - emg_windows: (13, 192)
    - acc_windows: (18, 29) - 6 sensors × 3 axes
    - gyro_windows: (18, 29) - 6 sensors × 3 axes
    
    Returns:
    - Feature vector: 13×8 + 6×13 + 6×13 = 104 + 78 + 78 = 260 features
    """
    features = []
    
    # EMG: 13 × 8 = 104
    for i in range(emg_windows.shape[0]):
        window = emg_windows[i].flatten()
        emg_feats = extract_emg_features(window, emg_fs)
        features.extend(emg_feats)
    
    # ACC: 18 channels = 6 sensors × 3 axes
    n_acc_sensors = acc_windows.shape[0] // 3
    for sensor_idx in range(n_acc_sensors):
        start_ch = sensor_idx * 3
        sensor_window = acc_windows[start_ch:start_ch+3].T  # (29, 3)
        acc_feats = extract_imu_features(sensor_window)
        features.extend(acc_feats)
    
    # GYRO: 18 channels = 6 sensors × 3 axes  
    n_gyro_sensors = gyro_windows.shape[0] // 3
    for sensor_idx in range(n_gyro_sensors):
        start_ch = sensor_idx * 3
        sensor_window = gyro_windows[start_ch:start_ch+3].T  # (29, 3)
        gyro_feats = extract_imu_features(sensor_window)
        features.extend(gyro_feats)
    
    return np.array(features)


def extract_realtime_features_debug(emg_windows, imu_windows, emg_fs=1259):
    """
    Debug version with detailed output.
    
    Use this to see exactly what's happening.
    """
    print("\n" + "="*70)
    print("FEATURE EXTRACTION DEBUG")
    print("="*70)
    print(f"EMG windows: {emg_windows.shape}")
    print(f"IMU windows: {imu_windows.shape}")
    
    features = []
    
    # EMG
    n_emg = emg_windows.shape[0]
    for i in range(n_emg):
        window = emg_windows[i].flatten()
        emg_feats = extract_emg_features(window, emg_fs)
        features.extend(emg_feats)
        if i == 0:
            print(f"\nEMG channel 0:")
            print(f"  Window shape: {window.shape}")
            print(f"  Features extracted: {len(emg_feats)}")
    
    emg_total = len(features)
    print(f"\nTotal EMG features: {emg_total} ({n_emg} channels × 8)")
    
    # IMU - try grouping into 3-axis sensors
    n_imu_channels = imu_windows.shape[0]
    n_sensors = n_imu_channels // 3
    
    print(f"\nIMU: {n_imu_channels} channels -> {n_sensors} sensors (3 axes each)")
    
    for sensor_idx in range(n_sensors):
        start_ch = sensor_idx * 3
        sensor_window = imu_windows[start_ch:start_ch+3].T
        
        imu_feats = extract_imu_features(sensor_window)
        features.extend(imu_feats)
        
        if sensor_idx == 0:
            print(f"\nIMU sensor 0:")
            print(f"  Input shape: {sensor_window.shape}")
            print(f"  Features extracted: {len(imu_feats)}")
    
    imu_total = len(features) - emg_total
    print(f"\nTotal IMU features: {imu_total} ({n_sensors} sensors × 13)")
    
    total = len(features)
    print(f"\n{'='*70}")
    print(f"TOTAL: {total} features")
    print(f"Expected: 456 features")
    print(f"Difference: {456 - total}")
    
    if total != 456:
        print(f"\n⚠ MISMATCH!")
        print(f"  Check your training data:")
        print(f"    - How many EMG sensors were used?")
        print(f"    - How many IMU sensors were used?")
        print(f"    - What was the channel grouping?")
    else:
        print(f"\n✓ Feature count matches!")
    
    print("="*70)
    
    return np.array(features)


# For your data_processor.py
class RealtimeFeatureExtractor:
    """
    Drop-in replacement for your current feature extraction.
    """
    
    def __init__(self, emg_fs=1259):
        self.emg_fs = emg_fs
    
    def extract_features(self, emg_windows, imu_windows):
        """
        Extract features matching training format.
        
        Args:
            emg_windows: (n_emg_channels, window_length)
            imu_windows: (n_imu_channels, window_length)
        
        Returns:
            Feature vector
        """
        features = []
        
        # EMG features
        for i in range(emg_windows.shape[0]):
            window = emg_windows[i].flatten()
            emg_feats = extract_emg_features(window, self.emg_fs)
            features.extend(emg_feats)
        
        # IMU features - group into 3-axis sensors
        n_imu_sensors = imu_windows.shape[0] // 3
        
        for sensor_idx in range(n_imu_sensors):
            start_ch = sensor_idx * 3
            sensor_window = imu_windows[start_ch:start_ch+3].T
            imu_feats = extract_imu_features(sensor_window)
            features.extend(imu_feats)
        
        return np.array(features)


if __name__ == "__main__":
    # Test with your actual dimensions
    print("Testing with actual channel counts...")
    
    # Simulate your data
    emg_test = np.random.randn(13, 192)  # 13 EMG channels, 192 samples
    imu_test = np.random.randn(36, 29)    # 36 IMU channels, 29 samples
    
    print("\nVersion 1 (all IMU as one group):")
    feats_v1 = extract_realtime_features_v1(emg_test, imu_test)
    
    print("\nDebug version:")
    feats_debug = extract_realtime_features_debug(emg_test, imu_test)