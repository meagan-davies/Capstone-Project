"""
Combine all feature extraction into single function
"""

import numpy as np
from typing import Dict
import sys
from pathlib import Path

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))
from control_metrics import (
    calculate_tracking_error,
    calculate_movement_smoothness,
    calculate_path_efficiency
)

from .physiological import extract_all_physiological_features
from .control_accuracy import extract_control_features


def extract_all_features(leap_data: Dict, 
                        bioradio_data: Dict, 
                        watch_data: Dict,
                        target_trajectory: np.ndarray = None) -> Dict:
    """
    Extract all features from multimodal sensor data
    
    Args:
        leap_data: Leap Motion data dict
        bioradio_data: BioRadio data dict
        watch_data: Apple Watch data dict
        target_trajectory: Optional target trajectory for tracking tasks
    
    Returns:
        Dict of all features (name -> value)
    """
    features = {}
    
    # 1. Control accuracy features (Leap Motion)
    control_features = extract_control_features(leap_data, target_trajectory)
    features.update({f"control_{k}": v for k, v in control_features.items()})
    
    # 2. Physiological features (BioRadio + Apple Watch)
    physio_features = extract_all_physiological_features(bioradio_data, watch_data)
    features.update(physio_features)
    
    # 3. Cross-sensor synchrony features
    synchrony_features = extract_synchrony_features(leap_data, bioradio_data, watch_data)
    features.update({f"sync_{k}": v for k, v in synchrony_features.items()})
    
    return features


def extract_synchrony_features(leap_data: Dict, 
                               bioradio_data: Dict, 
                               watch_data: Dict) -> Dict:
    """
    Extract cross-sensor synchrony features
    
    These measure how well different sensors are synchronized,
    which is important for embodiment
    
    Args:
        leap_data: Leap Motion data
        bioradio_data: BioRadio data
        watch_data: Apple Watch data
    
    Returns:
        Dict of synchrony features
    """
    features = {}
    
    # 1. EMG-Movement correlation
    if 'hand_position' in leap_data and 'emg' in bioradio_data:
        hand_velocity = np.linalg.norm(np.diff(leap_data['hand_position'], axis=0), axis=1)
        emg_envelope = np.mean(np.abs(bioradio_data['emg']), axis=1)
        
        # Resample to match lengths
        min_len = min(len(hand_velocity), len(emg_envelope))
        hand_velocity = hand_velocity[:min_len]
        emg_envelope = emg_envelope[:min_len]
        
        # Calculate correlation
        if len(hand_velocity) > 2:
            correlation = np.corrcoef(hand_velocity, emg_envelope)[0, 1]
            features['emg_movement_correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            features['emg_movement_correlation'] = 0.0
    
    # 2. Watch-Leap motion correlation
    if 'accelerometer' in watch_data and watch_data['accelerometer'] is not None:
        watch_accel_mag = np.linalg.norm(watch_data['accelerometer'], axis=1)
        hand_accel = np.linalg.norm(np.diff(np.diff(leap_data['hand_position'], axis=0), axis=0), axis=1)
        
        # Resample to match
        min_len = min(len(watch_accel_mag), len(hand_accel))
        watch_accel_mag = watch_accel_mag[:min_len]
        hand_accel = hand_accel[:min_len]
        
        if len(watch_accel_mag) > 2:
            correlation = np.corrcoef(watch_accel_mag, hand_accel)[0, 1]
            features['watch_leap_correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            features['watch_leap_correlation'] = 0.0
    
    # 3. HR-Movement correlation (arousal during movement)
    if 'heart_rate' in watch_data:
        hand_speed = np.linalg.norm(np.diff(leap_data['hand_position'], axis=0), axis=1)
        hr = watch_data['heart_rate']
        
        min_len = min(len(hand_speed), len(hr))
        hand_speed = hand_speed[:min_len]
        hr = hr[:min_len]
        
        if len(hand_speed) > 2:
            correlation = np.corrcoef(hand_speed, hr)[0, 1]
            features['hr_movement_correlation'] = correlation if not np.isnan(correlation) else 0.0
        else:
            features['hr_movement_correlation'] = 0.0
    
    # 4. Cross-correlation lag (EMG to movement)
    if 'hand_position' in leap_data and 'emg' in bioradio_data:
        hand_velocity = np.linalg.norm(np.diff(leap_data['hand_position'], axis=0), axis=1)
        emg_envelope = np.mean(np.abs(bioradio_data['emg']), axis=1)
        
        min_len = min(len(hand_velocity), len(emg_envelope))
        hand_velocity = hand_velocity[:min_len]
        emg_envelope = emg_envelope[:min_len]
        
        # Calculate cross-correlation
        if len(hand_velocity) > 10:
            cross_corr = np.correlate(emg_envelope, hand_velocity, mode='full')
            lag = np.argmax(cross_corr) - len(hand_velocity)
            features['emg_movement_lag_samples'] = abs(lag)
            
            # Convert to ms (assuming 100 Hz after synchronization)
            features['emg_movement_lag_ms'] = abs(lag) * 10
        else:
            features['emg_movement_lag_samples'] = 0
            features['emg_movement_lag_ms'] = 0
    
    return features


def get_feature_groups(feature_names: list) -> Dict[str, list]:
    """
    Group features by type for analysis
    
    Args:
        feature_names: List of all feature names
    
    Returns:
        Dict mapping group names to feature indices
    """
    groups = {
        'control': [],
        'cardiac': [],
        'muscle': [],
        'arousal': [],
        'synchrony': [],
        'motion': []
    }
    
    for i, name in enumerate(feature_names):
        if 'control' in name or 'tracking' in name or 'smoothness' in name:
            groups['control'].append(i)
        elif 'hr' in name or 'hrv' in name or 'ecg' in name:
            groups['cardiac'].append(i)
        elif 'emg' in name:
            groups['muscle'].append(i)
        elif 'eda' in name or 'gsr' in name:
            groups['arousal'].append(i)
        elif 'sync' in name or 'correlation' in name or 'lag' in name:
            groups['synchrony'].append(i)
        elif 'accel' in name or 'gyro' in name:
            groups['motion'].append(i)
    
    return groups


def filter_features_by_variance(X: np.ndarray, 
                                feature_names: list, 
                                threshold: float = 0.01) -> tuple:
    """
    Remove low-variance features
    
    Args:
        X: Feature matrix
        feature_names: List of feature names
        threshold: Minimum variance threshold
    
    Returns:
        Tuple of (filtered_X, filtered_feature_names)
    """
    variances = np.var(X, axis=0)
    high_var_mask = variances > threshold
    
    X_filtered = X[:, high_var_mask]
    filtered_names = [name for name, keep in zip(feature_names, high_var_mask) if keep]
    
    n_removed = len(feature_names) - len(filtered_names)
    print(f"Removed {n_removed} low-variance features (threshold={threshold})")
    
    return X_filtered, filtered_names