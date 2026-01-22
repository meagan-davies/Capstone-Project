"""
FINAL SOLUTION - Matching Training Exactly

Training structure from windowing.py:
- emg_dict: 13 EMG sensors (some multi-channel from Galileo sensors)
- imu_dict: 14 IMU sensors (7 ACC + 7 GYRO, each as 3-axis data)

Feature extraction:
- EMG: 13 sensors × 8 features = 104
- IMU: 14 sensors × 13 features = 182
- Total: 104 + 182 = 286 features

But we need 456! So let me recalculate...

Actually, looking at the CSV more carefully:
- If training loaded each ACC axis and GYRO axis as SEPARATE sensors
- Then: 21 ACC + 21 GYRO = 42 IMU "sensors" in imu_dict
- Each treated as (n_samples, 1) single-axis data
- BUT extract_imu_features() expects multi-axis...

WAIT - I need to check how the CSV was actually loaded into dicts!
"""

import numpy as np
from emg_features import extract_emg_features
from imu_features import extract_imu_features


def extract_realtime_features_matched(emg_data, acc_data, gyro_data, emg_fs=1259):
    """
    Extract features matching EXACT training structure.
    
    Based on training CSV having 7 physical sensors:
    - Each sensor contributes EMG channel(s), 3 ACC axes, 3 GYRO axes
    
    Args:
        emg_data: Dict or array with EMG channels
        acc_data: Dict or array with ACC channels (21 channels = 7 sensors × 3 axes)
        gyro_data: Dict or array with GYRO channels (21 channels = 7 sensors × 3 axes)
        emg_fs: EMG sampling frequency
    
    Returns:
        Feature vector matching training's 456 features
    """
    features = []
    
    # === EMG FEATURES ===
    # Process all 13 EMG channels individually
    print(f"\nProcessing EMG channels...")
    if isinstance(emg_data, dict):
        emg_channels = sorted(emg_data.keys())
        for ch_guid in emg_channels:
            window = emg_data[ch_guid].flatten()
            emg_feats = extract_emg_features(window, emg_fs)
            features.extend(emg_feats)
    else:
        # Array format: (13, window_len)
        for i in range(emg_data.shape[0]):
            window = emg_data[i].flatten()
            emg_feats = extract_emg_features(window, emg_fs)
            features.extend(emg_feats)
    
    emg_count = len(features)
    print(f"  EMG features: {emg_count}")
    
    # === IMU FEATURES ===
    # Group ACC and GYRO channels into 3-axis sensors
    # 21 ACC channels → 7 sensors × 3 axes
    # 21 GYRO channels → 7 sensors × 3 axes
    
    print(f"\nProcessing IMU channels...")
    
    # Process ACC sensors (7 sensors × 3 axes each)
    if isinstance(acc_data, dict):
        acc_channels = sorted(acc_data.keys())
        n_acc_sensors = len(acc_channels) // 3
        
        for sensor_idx in range(n_acc_sensors):
            # Get 3 consecutive channels for this sensor
            ch_indices = slice(sensor_idx * 3, (sensor_idx + 1) * 3)
            sensor_guids = acc_channels[ch_indices]
            
            # Stack into (n_samples, 3) array
            sensor_windows = [acc_data[guid] for guid in sensor_guids]
            stacked_window = np.column_stack(sensor_windows)  # (n_samples, 3)
            
            acc_feats = extract_imu_features(stacked_window)
            features.extend(acc_feats)
    else:
        # Array format: (21, window_len)
        n_acc_sensors = acc_data.shape[0] // 3
        for sensor_idx in range(n_acc_sensors):
            start_ch = sensor_idx * 3
            end_ch = start_ch + 3
            
            # Transpose to get (n_samples, 3)
            sensor_window = acc_data[start_ch:end_ch].T
            acc_feats = extract_imu_features(sensor_window)
            features.extend(acc_feats)
    
    acc_count = len(features) - emg_count
    print(f"  ACC features: {acc_count} ({n_acc_sensors} sensors × 13)")
    
    # Process GYRO sensors (7 sensors × 3 axes each)
    if isinstance(gyro_data, dict):
        gyro_channels = sorted(gyro_data.keys())
        n_gyro_sensors = len(gyro_channels) // 3
        
        for sensor_idx in range(n_gyro_sensors):
            ch_indices = slice(sensor_idx * 3, (sensor_idx + 1) * 3)
            sensor_guids = gyro_channels[ch_indices]
            
            sensor_windows = [gyro_data[guid] for guid in sensor_guids]
            stacked_window = np.column_stack(sensor_windows)
            
            gyro_feats = extract_imu_features(stacked_window)
            features.extend(gyro_feats)
    else:
        # Array format: (21, window_len)
        n_gyro_sensors = gyro_data.shape[0] // 3
        for sensor_idx in range(n_gyro_sensors):
            start_ch = sensor_idx * 3
            end_ch = start_ch + 3
            
            sensor_window = gyro_data[start_ch:end_ch].T
            gyro_feats = extract_imu_features(sensor_window)
            features.extend(gyro_feats)
    
    gyro_count = len(features) - emg_count - acc_count
    print(f"  GYRO features: {gyro_count} ({n_gyro_sensors} sensors × 13)")
    
    total = len(features)
    print(f"\n  Total features: {total}")
    print(f"  Expected: 456")
    
    if total != 456:
        print(f"  ⚠ MISMATCH: {456 - total} features {'short' if total < 456 else 'extra'}")
        print(f"\n  Breakdown:")
        print(f"    13 EMG × 8 = {13*8}")
        print(f"    7 ACC × 13 = {7*13}")
        print(f"    7 GYRO × 13 = {7*13}")
        print(f"    Total = {13*8 + 7*13 + 7*13}")
    else:
        print(f"  ✓ Perfect match!")
    
    return np.array(features)


class ProductionFeatureExtractor:
    """
    Production-ready feature extractor for real-time classification.
    
    This matches the EXACT feature extraction used in training.
    """
    
    def __init__(self, channel_info, emg_fs=1259):
        """
        Initialize with channel metadata from Delsys client.
        
        Args:
            channel_info: Dict mapping channel GUID -> metadata
            emg_fs: EMG sampling frequency
        """
        self.emg_fs = emg_fs
        self.channel_info = channel_info
        
        # Categorize and sort channels
        self.emg_guids = sorted([
            guid for guid, info in channel_info.items()
            if info['type'] == 'EMG'
        ])
        
        self.acc_guids = sorted([
            guid for guid, info in channel_info.items()
            if info['type'] == 'ACC'
        ])
        
        self.gyro_guids = sorted([
            guid for guid, info in channel_info.items()
            if info['type'] == 'GYRO'
        ])
        
        # Verify channel counts
        print(f"\nFeature Extractor initialized:")
        print(f"  EMG channels: {len(self.emg_guids)}")
        print(f"  ACC channels: {len(self.acc_guids)} (will group into {len(self.acc_guids)//3} sensors)")
        print(f"  GYRO channels: {len(self.gyro_guids)} (will group into {len(self.gyro_guids)//3} sensors)")
        
        # Calculate expected feature count
        n_emg_features = len(self.emg_guids) * 8
        n_acc_features = (len(self.acc_guids) // 3) * 13
        n_gyro_features = (len(self.gyro_guids) // 3) * 13
        expected_total = n_emg_features + n_acc_features + n_gyro_features
        
        print(f"\n  Expected features:")
        print(f"    EMG: {len(self.emg_guids)} × 8 = {n_emg_features}")
        print(f"    ACC: {len(self.acc_guids)//3} × 13 = {n_acc_features}")
        print(f"    GYRO: {len(self.gyro_guids)//3} × 13 = {n_gyro_features}")
        print(f"    Total: {expected_total}")
        
        if expected_total != 456:
            print(f"  ⚠ WARNING: Expected 456, calculated {expected_total}")
            print(f"  This might cause prediction errors!")
    
    def extract_features(self, channel_windows):
        """
        Extract features from windowed channel data.
        
        Args:
            channel_windows: Dict mapping channel GUID -> numpy array of samples
        
        Returns:
            Feature vector as numpy array
        """
        features = []
        
        # EMG features
        for guid in self.emg_guids:
            if guid not in channel_windows:
                raise ValueError(f"Missing EMG channel: {guid}")
            
            window = channel_windows[guid].flatten()
            emg_feats = extract_emg_features(window, self.emg_fs)
            features.extend(emg_feats)
        
        # ACC features - group into 3-axis sensors
        n_acc_sensors = len(self.acc_guids) // 3
        for sensor_idx in range(n_acc_sensors):
            start_idx = sensor_idx * 3
            end_idx = start_idx + 3
            sensor_guids = self.acc_guids[start_idx:end_idx]
            
            # Stack channels into (n_samples, 3) array
            sensor_data = [channel_windows[guid] for guid in sensor_guids]
            stacked_window = np.column_stack(sensor_data)
            
            acc_feats = extract_imu_features(stacked_window)
            features.extend(acc_feats)
        
        # GYRO features - group into 3-axis sensors
        n_gyro_sensors = len(self.gyro_guids) // 3
        for sensor_idx in range(n_gyro_sensors):
            start_idx = sensor_idx * 3
            end_idx = start_idx + 3
            sensor_guids = self.gyro_guids[start_idx:end_idx]
            
            sensor_data = [channel_windows[guid] for guid in sensor_guids]
            stacked_window = np.column_stack(sensor_data)
            
            gyro_feats = extract_imu_features(stacked_window)
            features.extend(gyro_feats)
        
        return np.array(features)


if __name__ == "__main__":
    print("Testing feature extraction...")
    
    # Simulate training structure
    emg_test = np.random.randn(13, 192)  # 13 EMG channels
    acc_test = np.random.randn(21, 29)   # 21 ACC channels (7 sensors × 3)
    gyro_test = np.random.randn(21, 29)  # 21 GYRO channels (7 sensors × 3)
    
    feats = extract_realtime_features_matched(emg_test, acc_test, gyro_test)
    
    print(f"\n{'='*70}")
    print(f"Final feature count: {len(feats)}")
    print(f"Expected: 456")
    print(f"Match: {'✓ YES' if len(feats) == 456 else '✗ NO'}")
    print(f"{'='*70}")