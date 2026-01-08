"""
Real-Time Data Processor

Handles:
- Buffering streaming data from Delsys sensors
- Managing different sampling rates (EMG vs IMU)
- Extracting features when windows are ready
"""

import numpy as np
from collections import deque
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
        """
        Initialize processor.
        
        Args:
            emg_channel_guids: List of EMG channel GUIDs
            imu_channel_guids: List of IMU channel GUIDs
            fs_emg: EMG sampling frequency (Hz)
            fs_imu: IMU sampling frequency (Hz)
            window_sec: Window duration (seconds)
            overlap_sec: Window overlap (seconds)
        """
        self.emg_channel_guids = sorted(emg_channel_guids)
        self.imu_channel_guids = sorted(imu_channel_guids)
        
        self.fs_emg = fs_emg
        self.fs_imu = fs_imu
        
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        
        # Calculate buffer sizes
        self.emg_win_size = int(window_sec * fs_emg)
        self.imu_win_size = int(window_sec * fs_imu)
        
        self.emg_step = int((window_sec - overlap_sec) * fs_emg)
        self.imu_step = int((window_sec - overlap_sec) * fs_imu)
        
        # Create circular buffers for each channel
        self.emg_buffers = {
            guid: deque(maxlen=self.emg_win_size * 2)
            for guid in emg_channel_guids
        }
        
        self.imu_buffers = {
            guid: deque(maxlen=self.imu_win_size * 2)
            for guid in imu_channel_guids
        }
        
        # Sample counters for step tracking
        self.emg_sample_count = 0
        self.imu_sample_count = 0
        
        # Thread safety
        self.lock = Lock()
        
        print(f"✓ Processor initialized")
        print(f"  EMG: {len(emg_channel_guids)} channels, {self.emg_win_size} samples/window")
        print(f"  IMU: {len(imu_channel_guids)} channels, {self.imu_win_size} samples/window")
    
    def add_data(self, data_dict: Dict[str, np.ndarray]):
        """
        Add new data samples from Delsys API.
        
        Args:
            data_dict: Dictionary mapping GUIDs to data arrays
        """
        with self.lock:
            for guid, values in data_dict.items():
                # Add to appropriate buffer
                if guid in self.emg_buffers:
                    for val in values:
                        self.emg_buffers[guid].append(val)
                    self.emg_sample_count += len(values)
                
                elif guid in self.imu_buffers:
                    for val in values:
                        self.imu_buffers[guid].append(val)
                    self.imu_sample_count += len(values)
    
    def is_window_ready(self) -> bool:
        """
        Check if we have enough data for a new window.
        
        Returns:
            True if ready to extract features
        """
        # Check if buffers have enough samples
        emg_ready = all(
            len(buf) >= self.emg_win_size 
            for buf in self.emg_buffers.values()
        )
        
        imu_ready = all(
            len(buf) >= self.imu_win_size 
            for buf in self.imu_buffers.values()
        )
        
        # Check if we've moved enough samples for next window
        emg_step_ready = self.emg_sample_count >= self.emg_step
        imu_step_ready = self.imu_sample_count >= self.imu_step
        
        return emg_ready and imu_ready and emg_step_ready and imu_step_ready
    
    def extract_window_features(self) -> Optional[np.ndarray]:
        """
        Extract features from current window.
        
        Uses the SAME feature extraction as training!
        
        Returns:
            Feature vector, or None if window not ready
        """
        with self.lock:
            if not self.is_window_ready():
                return None
            
            # Get windows from buffers
            emg_windows = {}
            for guid in self.emg_channel_guids:
                buf = self.emg_buffers[guid]
                emg_windows[guid] = np.array(list(buf)[-self.emg_win_size:])
            
            imu_windows = {}
            for guid in self.imu_channel_guids:
                buf = self.imu_buffers[guid]
                imu_windows[guid] = np.array(list(buf)[-self.imu_win_size:])
            
            # Reset step counters
            self.emg_sample_count = 0
            self.imu_sample_count = 0
        
        # Extract features (outside lock for performance)
        features = []
        
        # EMG features - one per channel
        for guid in self.emg_channel_guids:
            signal = emg_windows[guid]
            emg_feats = extract_emg_features(signal, fs=self.fs_emg)
            features.extend(emg_feats)
        
        # IMU features - combine all IMU channels
        if len(imu_windows) > 0:
            # Stack all IMU channels
            imu_stacked = np.column_stack([
                imu_windows[guid] for guid in self.imu_channel_guids
            ])
            imu_feats = extract_imu_features(imu_stacked)
            features.extend(imu_feats)
        
        return np.array(features)
    
    def get_buffer_status(self) -> Dict:
        """Get current buffer status (for debugging)"""
        with self.lock:
            return {
                'emg_buffer_sizes': {
                    guid: len(buf) for guid, buf in self.emg_buffers.items()
                },
                'imu_buffer_sizes': {
                    guid: len(buf) for guid, buf in self.imu_buffers.items()
                },
                'emg_sample_count': self.emg_sample_count,
                'imu_sample_count': self.imu_sample_count,
                'window_ready': self.is_window_ready()
            }


if __name__ == "__main__":
    print("Testing processor...")
    
    # Create fake GUIDs
    emg_guids = [f"emg-{i}" for i in range(4)]
    imu_guids = [f"imu-{i}" for i in range(24)]
    
    processor = RealtimeProcessor(emg_guids, imu_guids)
    
    # Simulate adding data
    fake_data = {
        **{guid: np.random.randn(10) for guid in emg_guids},
        **{guid: np.random.randn(2) for guid in imu_guids}
    }
    
    # Add data until window ready
    for i in range(30):
        processor.add_data(fake_data)
        status = processor.get_buffer_status()
        print(f"Step {i}: Window ready? {status['window_ready']}")
        
        if status['window_ready']:
            features = processor.extract_window_features()
            print(f"✓ Extracted {len(features)} features!")
            break
    
    print("\n✓ Processor test complete!")