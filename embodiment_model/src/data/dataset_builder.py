"""
Build aligned feature matrices from multimodal sessions
"""

import numpy as np
import sys
from pathlib import Path
from typing import List, Tuple, Dict

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))
from synchronization import synchronize_streams

from .data_loader import EmbodimentSession


def synchronize_session_data(session: EmbodimentSession, target_fs: int = 100) -> Dict:
    """
    Synchronize all sensor streams to common sampling rate
    
    Args:
        session: EmbodimentSession object
        target_fs: Target sampling frequency (Hz)
    
    Returns:
        Dict with synchronized data
    """
    # Prepare streams dict for synchronization
    streams = {
        'leap': {
            'data': session.leap_data['hand_position'],
            'timestamps': session.leap_data['timestamps'],
            'fs': 115
        },
        'bioradio_emg': {
            'data': session.bioradio_data['emg'],
            'timestamps': session.bioradio_data['timestamps'],
            'fs': session.bioradio_data['sampling_rate']
        },
        'bioradio_ecg': {
            'data': session.bioradio_data['ecg'],
            'timestamps': session.bioradio_data['timestamps'],
            'fs': session.bioradio_data['sampling_rate']
        },
        'bioradio_eda': {
            'data': session.bioradio_data['eda'],
            'timestamps': session.bioradio_data['timestamps'],
            'fs': session.bioradio_data['sampling_rate']
        },
        'watch_hr': {
            'data': session.watch_data['heart_rate'],
            'timestamps': session.watch_data['timestamps'],
            'fs': session.watch_data['sampling_rate']
        }
    }
    
    # Add accelerometer if available
    if session.watch_data['accelerometer'] is not None:
        streams['watch_accel'] = {
            'data': session.watch_data['accelerometer'],
            'timestamps': session.watch_data['timestamps'],
            'fs': session.watch_data['sampling_rate']
        }
    
    # Synchronize
    synchronized = synchronize_streams(streams, target_fs=target_fs)
    
    return synchronized


def build_feature_matrix(sessions: List[EmbodimentSession],
                        feature_extractor=None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Build complete feature matrix from sessions
    
    Args:
        sessions: List of EmbodimentSession objects
        feature_extractor: Optional custom feature extraction function
    
    Returns:
        Tuple of (X, y, participant_ids, feature_names)
        - X: Feature matrix (n_sessions, n_features)
        - y: Embodiment scores (n_sessions,)
        - participant_ids: Participant IDs (n_sessions,)
        - feature_names: List of feature names
    """
    if feature_extractor is None:
        from ..features.feature_stack import extract_all_features
        feature_extractor = extract_all_features
    
    X_list = []
    y_list = []
    participant_ids = []
    feature_names = None
    
    print(f"Building feature matrix from {len(sessions)} sessions...")
    
    for i, session in enumerate(sessions):
        try:
            # Synchronize sensor data
            sync_data = synchronize_session_data(session)
            
            # Extract features
            features_dict = feature_extractor(
                leap_data={
                    'hand_position': sync_data['leap'],
                    'timestamps': sync_data['timestamps']
                },
                bioradio_data={
                    'emg': sync_data['bioradio_emg'],
                    'ecg': sync_data['bioradio_ecg'],
                    'eda': sync_data['bioradio_eda'],
                    'sampling_rate': 100
                },
                watch_data={
                    'heart_rate': sync_data['watch_hr'],
                    'accelerometer': sync_data.get('watch_accel'),
                    'sampling_rate': 100
                }
            )
            
            # Store feature names from first session
            if feature_names is None:
                feature_names = list(features_dict.keys())
            
            # Convert to array
            feature_values = [features_dict[name] for name in feature_names]
            X_list.append(feature_values)
            
            # Store label
            y_list.append(session.embodiment_score)
            
            # Store participant ID
            participant_ids.append(session.participant_id)
            
            if (i + 1) % 10 == 0:
                print(f"  Processed {i + 1}/{len(sessions)} sessions")
            
        except Exception as e:
            print(f"⚠ Error processing session {i}: {e}")
            continue
    
    # Convert to numpy arrays
    X = np.array(X_list)
    y = np.array(y_list)
    participant_ids = np.array(participant_ids)
    
    print(f"✓ Feature matrix built: {X.shape}")
    print(f"  Features: {len(feature_names)}")
    print(f"  Sessions: {len(y)}")
    print(f"  Participants: {len(np.unique(participant_ids))}")
    
    return X, y, participant_ids, feature_names


def split_by_participant(X: np.ndarray, 
                        y: np.ndarray, 
                        participant_ids: np.ndarray,
                        test_participants: List[str]) -> Tuple:
    """
    Split data by participant (for train/test split)
    
    Args:
        X: Feature matrix
        y: Labels
        participant_ids: Participant IDs
        test_participants: List of participant IDs for test set
    
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, train_ids, test_ids)
    """
    test_mask = np.isin(participant_ids, test_participants)
    train_mask = ~test_mask
    
    X_train = X[train_mask]
    X_test = X[test_mask]
    y_train = y[train_mask]
    y_test = y[test_mask]
    train_ids = participant_ids[train_mask]
    test_ids = participant_ids[test_mask]
    
    print(f"Train set: {len(y_train)} samples from {len(np.unique(train_ids))} participants")
    print(f"Test set: {len(y_test)} samples from {len(np.unique(test_ids))} participants")
    
    return X_train, X_test, y_train, y_test, train_ids, test_ids


def add_interaction_features(X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Add interaction features between control accuracy and physiological signals
    
    Args:
        X: Feature matrix
        feature_names: List of feature names
    
    Returns:
        Tuple of (X_with_interactions, new_feature_names)
    """
    # Find control and physio feature indices
    control_indices = [i for i, name in enumerate(feature_names) 
                      if any(keyword in name for keyword in ['tracking', 'smoothness', 'latency'])]
    
    physio_indices = [i for i, name in enumerate(feature_names)
                     if any(keyword in name for keyword in ['hrv', 'emg', 'eda', 'hr'])]
    
    # Create interaction features
    interactions = []
    interaction_names = []
    
    for control_idx in control_indices:
        for physio_idx in physio_indices:
            # Multiplicative interaction
            interaction = X[:, control_idx] * X[:, physio_idx]
            interactions.append(interaction)
            interaction_names.append(f"{feature_names[control_idx]}_x_{feature_names[physio_idx]}")
    
    # Concatenate
    if interactions:
        X_interactions = np.column_stack(interactions)
        X_augmented = np.hstack([X, X_interactions])
        new_feature_names = feature_names + interaction_names
        
        print(f"✓ Added {len(interaction_names)} interaction features")
        return X_augmented, new_feature_names
    
    return X, feature_names