"""
Real-Time Data Processor

Handles:
- Buffering streaming data from Delsys sensors
- Managing different sampling rates (EMG vs IMU)
- Extracting features when windows are ready
"""

import numpy as np
from collections import deque, defaultdict
from threading import Lock
from typing import Dict, List, Optional

# Import your feature extraction functions
import sys
from pathlib import Path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from features.emg_features import extract_emg_features
from features.imu_features import extract_imu_features


def group_imu_by_sensor(imu_channel_guids: List[str]) -> Dict[str, List[str]]:
    """
    Group flat IMU channel list into sensors.
    Example: ACC-1-X, GYRO-1-Y → sensor '1'
    """
    sensor_groups = defaultdict(list)
    for guid in imu_channel_guids:
        # Extract sensor ID assuming format TYPE-<sensor_id>-AXIS
        parts = guid.split('-')
        if len(parts) >= 3:
            sensor_id = parts[1]
            sensor_groups[sensor_id].append(guid)
    return dict(sensor_groups)

def group_emg_by_sensor(emg_channel_guids: List[str]) -> Dict[str, List[str]]:
    """
    Group Galileo EMG channels into sensors.
    Example: EMG-1-1, EMG-1-2, EMG-1-3, EMG-1-4 → sensor '1'
    """
    sensor_groups = defaultdict(list)
    for guid in emg_channel_guids:
        parts = guid.split('-')
        if len(parts) >= 3:
            sensor_id = parts[1]
            sensor_groups[sensor_id].append(guid)
    return dict(sensor_groups)

class RealtimeProcessor:
    """
    Processes streaming EMG and IMU data in real-time.
    
    Manages circular buffers, windowing, and feature extraction
    to prepare data for classification.
    """
    
    def __init__(
        self,
        emg_channel_guids: List[str],
        imu_channel_guids: List[str],
        fs_emg: float = 963,
        fs_imu: float = 148.148,
        window_sec: float = 0.20,
        overlap_sec: float = 0.10
    ):
        self.emg_channel_guids = sorted(emg_channel_guids)
        self.imu_channel_guids = sorted(imu_channel_guids)
        self.fs_emg = fs_emg
        self.fs_imu = fs_imu
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        
        # Window sizes and steps
        self.emg_win_size = int(window_sec * fs_emg)
        self.imu_win_size = int(window_sec * fs_imu)
        self.emg_step = int((window_sec - overlap_sec) * fs_emg)
        self.imu_step = int((window_sec - overlap_sec) * fs_imu)
        
        # Circular buffers
        self.emg_buffers = {guid: deque(maxlen=self.emg_win_size*2) for guid in emg_channel_guids}
        self.imu_buffers = {guid: deque(maxlen=self.imu_win_size*2) for guid in imu_channel_guids}
        
        # Sample counters
        self.emg_sample_count = 0
        self.imu_sample_count = 0
        
        # Thread safety
        self.lock = Lock()
        
        # Group IMU channels per sensor (matches training data)
        self.imu_sensor_map = group_imu_by_sensor(imu_channel_guids)

        # Group EMG channels per Galileo sensor (matches training data)
        self.emg_sensor_map = group_emg_by_sensor(emg_channel_guids)

        print(f"✓ Processor initialized")
        print(f"  EMG: {len(self.emg_sensor_map)} sensors (Galileo), "
              f"{self.emg_win_size} samples/window per sensor")
        print(f"  IMU: {len(self.imu_sensor_map)} sensors, {self.imu_win_size} samples/window per sensor")
    
    def add_data(self, data_dict: Dict[str, np.ndarray]):
        """Add new data samples from Delsys API."""
        with self.lock:
            for guid, values in data_dict.items():
                if guid in self.emg_buffers:
                    self.emg_buffers[guid].extend(values)
                    self.emg_sample_count += len(values)
                elif guid in self.imu_buffers:
                    self.imu_buffers[guid].extend(values)
                    self.imu_sample_count += len(values)
    
    def is_window_ready(self) -> bool:
        """Check if we have enough data for a new window."""
        emg_ready = all(len(buf) >= self.emg_win_size for buf in self.emg_buffers.values())
        imu_ready = all(len(buf) >= self.imu_win_size for buf in self.imu_buffers.values())
        emg_step_ready = self.emg_sample_count >= self.emg_step
        imu_step_ready = self.imu_sample_count >= self.imu_step
        return emg_ready and imu_ready and emg_step_ready and imu_step_ready
    
    def extract_window_features(self) -> Optional[np.ndarray]:
        """Extract features from current window."""
        with self.lock:
            if not self.is_window_ready():
                return None
            
            # Get EMG windows
            emg_windows = {guid: np.array(list(buf)[-self.emg_win_size:])
                           for guid, buf in self.emg_buffers.items()}
            
            # Get IMU windows
            imu_windows = {guid: np.array(list(buf)[-self.imu_win_size:])
                           for guid, buf in self.imu_buffers.items()}
            
            # Reset step counters
            self.emg_sample_count = 0
            self.imu_sample_count = 0
        
        # Extract features
        features = []
        
        # ======================
        # EMG features (Galileo stacked — MATCHES TRAINING)
        # ======================
        print("🧠 EMG feature sources:")
        for sensor_id, channels in sorted(self.emg_sensor_map.items()):
            # Stack EMG channels like training loader
            sensor_emg = np.column_stack([
                emg_windows[guid] for guid in sorted(channels)
            ])  # shape (N, n_channels)

            emg_feats = extract_emg_features(sensor_emg, fs=self.fs_emg)

            print(
                f"🧠 EMG Sensor {sensor_id}: "
                f"{len(emg_feats)} features → "
                f"{np.round(emg_feats, 3)}"
            )

            features.extend(emg_feats)


            print(f"  Sensor {sensor_id}: {sensor_emg.shape[1]} EMG channels → 1 feature block")

        # IMU features per sensor
        for sensor_id, channels in self.imu_sensor_map.items():
            sensor_data = np.column_stack([imu_windows[guid] for guid in channels])
            features.extend(extract_imu_features(sensor_data))
        print(f"✅ IMU sensors used for features: {list(self.imu_sensor_map.keys())}")
        
        return np.array(features)
    
    def get_buffer_status(self) -> Dict:
        """Get current buffer status (for debugging)."""
        with self.lock:
            return {
                'emg_buffer_sizes': {guid: len(buf) for guid, buf in self.emg_buffers.items()},
                'imu_buffer_sizes': {guid: len(buf) for guid, buf in self.imu_buffers.items()},
                'emg_sample_count': self.emg_sample_count,
                'imu_sample_count': self.imu_sample_count,
                'window_ready': self.is_window_ready()
            }


if __name__ == "__main__":
    print("Testing processor...")
    
    # Example fake GUIDs
    emg_guids = [f"emg-{i}" for i in range(7)]
    imu_guids = [f"ACC-{i}-{axis}" for i in range(1, 8) for axis in ['X','Y','Z']] + \
                [f"GYRO-{i}-{axis}" for i in range(1, 8) for axis in ['X','Y','Z']]
    
    processor = RealtimeProcessor(emg_guids, imu_guids)
    
    # Simulate adding data
    fake_data = {guid: np.random.randn(10) for guid in emg_guids + imu_guids}
    
    for i in range(30):
        processor.add_data(fake_data)
        status = processor.get_buffer_status()
        print(f"Step {i}: Window ready? {status['window_ready']}")
        if status['window_ready']:
            feats = processor.extract_window_features()
            print(f"✓ Extracted {len(feats)} features!")
            break
    
    print("\n✓ Processor test complete!")
