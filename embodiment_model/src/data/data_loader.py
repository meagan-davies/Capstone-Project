"""
Load multimodal embodiment data from disk
"""

import numpy as np
import pandas as pd
import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

# TODO: fix these functions with proper data loading examples and test
@dataclass
class EmbodimentSession:
    """Single embodiment recording session"""
    participant_id: str
    condition: str
    trial_number: int
    
    # Sensor data
    leap_data: Dict
    bioradio_data: Dict
    watch_data: Dict
    
    # Ground truth
    embodiment_score: float
    veq_scores: Optional[Dict] = None
    
    # Metadata
    timestamp: Optional[str] = None
    duration: Optional[float] = None


def load_embodiment_session(session_dir: Path) -> EmbodimentSession:
    """
    Load a single embodiment session
    
    Args:
        session_dir: Directory containing trial files
    
    Returns:
        EmbodimentSession object
    """
    # Load labels
    label_file = session_dir / f"{session_dir.name}_labels.json"
    with open(label_file, 'r') as f:
        labels = json.load(f)
    
    # Load sensor data
    leap_data = pd.read_csv(session_dir / f"{session_dir.name}_leap.csv")
    bioradio_data = pd.read_csv(session_dir / f"{session_dir.name}_bioradio.csv")
    watch_data = pd.read_csv(session_dir / f"{session_dir.name}_watch.csv")
    
    # Convert to dict format
    leap_dict = {
        'timestamps': leap_data['timestamp'].values,
        'hand_position': leap_data[['x', 'y', 'z']].values,
        'hand_velocity': leap_data[['vx', 'vy', 'vz']].values if 'vx' in leap_data.columns else None,
        'tracking_confidence': leap_data['confidence'].values if 'confidence' in leap_data.columns else None
    }
    
    bioradio_dict = {
        'timestamps': bioradio_data['timestamp'].values,
        'emg': bioradio_data[[f'emg_{i}' for i in range(4)]].values,
        'ecg': bioradio_data['ecg'].values,
        'eda': bioradio_data['eda'].values,
        'sampling_rate': 1000
    }
    
    watch_dict = {
        'timestamps': watch_data['timestamp'].values,
        'heart_rate': watch_data['heart_rate'].values,
        'accelerometer': watch_data[['accel_x', 'accel_y', 'accel_z']].values if 'accel_x' in watch_data.columns else None,
        'sampling_rate': 50
    }
    
    session = EmbodimentSession(
        participant_id=labels['participant_id'],
        condition=labels['condition'],
        trial_number=labels['trial_number'],
        leap_data=leap_dict,
        bioradio_data=bioradio_dict,
        watch_data=watch_dict,
        embodiment_score=labels['embodiment_score'],
        veq_scores={
            'ownership': labels.get('veq_ownership'),
            'agency': labels.get('veq_agency'),
            'location': labels.get('veq_location')
        },
        timestamp=labels.get('timestamp')
    )
    
    return session


def load_embodiment_sessions(data_dir: Path, 
                            participant_ids: Optional[List[str]] = None,
                            conditions: Optional[List[str]] = None) -> List[EmbodimentSession]:
    """
    Load multiple embodiment sessions
    
    Args:
        data_dir: Root data directory
        participant_ids: Filter by participant IDs (None = all)
        conditions: Filter by conditions (None = all)
    
    Returns:
        List of EmbodimentSession objects
    """
    data_dir = Path(data_dir)
    sessions = []
    
    # Recursively find all session directories
    for session_dir in data_dir.rglob('trial_*'):
        if not session_dir.is_dir():
            continue
        
        # Check if all required files exist
        required_files = [
            f"{session_dir.name}_leap.csv",
            f"{session_dir.name}_bioradio.csv",
            f"{session_dir.name}_watch.csv",
            f"{session_dir.name}_labels.json"
        ]
        
        if not all((session_dir / f).exists() for f in required_files):
            print(f"⚠ Skipping {session_dir} - missing files")
            continue
        
        try:
            session = load_embodiment_session(session_dir)
            
            # Apply filters
            if participant_ids and session.participant_id not in participant_ids:
                continue
            if conditions and session.condition not in conditions:
                continue
            
            sessions.append(session)
            
        except Exception as e:
            print(f"⚠ Error loading {session_dir}: {e}")
            continue
    
    print(f"✓ Loaded {len(sessions)} sessions")
    return sessions


def save_embodiment_session(session: EmbodimentSession, output_dir: Path) -> None:
    """
    Save embodiment session to disk
    
    Args:
        session: EmbodimentSession to save
        output_dir: Output directory
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    trial_name = f"trial_{session.trial_number:03d}"
    
    # Save Leap Motion data
    leap_df = pd.DataFrame({
        'timestamp': session.leap_data['timestamps'],
        'x': session.leap_data['hand_position'][:, 0],
        'y': session.leap_data['hand_position'][:, 1],
        'z': session.leap_data['hand_position'][:, 2],
    })
    if session.leap_data['hand_velocity'] is not None:
        leap_df['vx'] = session.leap_data['hand_velocity'][:, 0]
        leap_df['vy'] = session.leap_data['hand_velocity'][:, 1]
        leap_df['vz'] = session.leap_data['hand_velocity'][:, 2]
    if session.leap_data['tracking_confidence'] is not None:
        leap_df['confidence'] = session.leap_data['tracking_confidence']
    
    leap_df.to_csv(output_dir / f"{trial_name}_leap.csv", index=False)
    
    # Save BioRadio data
    bioradio_df = pd.DataFrame({
        'timestamp': session.bioradio_data['timestamps'],
        'emg_0': session.bioradio_data['emg'][:, 0],
        'emg_1': session.bioradio_data['emg'][:, 1],
        'emg_2': session.bioradio_data['emg'][:, 2],
        'emg_3': session.bioradio_data['emg'][:, 3],
        'ecg': session.bioradio_data['ecg'],
        'eda': session.bioradio_data['eda'],
    })
    bioradio_df.to_csv(output_dir / f"{trial_name}_bioradio.csv", index=False)
    
    # Save Apple Watch data
    watch_df = pd.DataFrame({
        'timestamp': session.watch_data['timestamps'],
        'heart_rate': session.watch_data['heart_rate'],
    })
    if session.watch_data['accelerometer'] is not None:
        watch_df['accel_x'] = session.watch_data['accelerometer'][:, 0]
        watch_df['accel_y'] = session.watch_data['accelerometer'][:, 1]
        watch_df['accel_z'] = session.watch_data['accelerometer'][:, 2]
    
    watch_df.to_csv(output_dir / f"{trial_name}_watch.csv", index=False)
    
    # Save labels
    labels = {
        'participant_id': session.participant_id,
        'condition': session.condition,
        'trial_number': session.trial_number,
        'embodiment_score': float(session.embodiment_score),
        'timestamp': session.timestamp
    }
    
    if session.veq_scores:
        labels['veq_ownership'] = session.veq_scores.get('ownership')
        labels['veq_agency'] = session.veq_scores.get('agency')
        labels['veq_location'] = session.veq_scores.get('location')
    
    with open(output_dir / f"{trial_name}_labels.json", 'w') as f:
        json.dump(labels, f, indent=2)
    
    print(f"✓ Saved session to {output_dir / trial_name}")