"""
Real-time embodiment monitoring
"""

import numpy as np
import time
from collections import deque
from typing import Optional
import sys
from pathlib import Path

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))
from sensors.leap_motion import LeapMotionCapture
from sensors.bioradio import BioRadioCapture
from sensors.apple_watch import AppleWatchCapture

from ..models.regressor import EmbodimentRegressor
from ..features.feature_stack import extract_all_features


class EmbodimentMonitor:
    """
    Real-time embodiment monitoring system
    """
    
    def __init__(self, 
                 model: EmbodimentRegressor,
                 window_size: float = 5.0,
                 update_rate: float = 10.0):
        """
        Initialize real-time monitor
        
        Args:
            model: Trained embodiment model
            window_size: Analysis window size (seconds)
            update_rate: Update frequency (Hz)
        """
        self.model = model
        self.window_size = window_size
        self.update_interval = 1.0 / update_rate
        
        # Initialize sensors
        self.leap = LeapMotionCapture()
        self.bioradio = BioRadioCapture()
        self.watch = AppleWatchCapture()
        
        # Buffered data
        self.buffer_size = int(window_size * 100)  # Assume 100 Hz after sync
        self.leap_buffer = deque(maxlen=self.buffer_size)
        self.bioradio_buffer = deque(maxlen=self.buffer_size)
        self.watch_buffer = deque(maxlen=self.buffer_size)
        
        # Embodiment score history
        self.score_history = deque(maxlen=100)  # Last 100 scores
        self.is_running = False
        
        print("✓ Embodiment Monitor initialized")
    
    def start(self):
        """Start real-time monitoring"""
        print("Starting embodiment monitoring...")
        
        self.is_running = True
        self.leap.start_recording()
        self.bioradio.start_recording()
        self.watch.start_recording()
        
        last_update = time.time()
        
        try:
            while self.is_running:
                current_time = time.time()
                
                # Collect sensor samples
                leap_sample = self.leap.get_frame()
                bioradio_sample = self.bioradio.get_sample()
                watch_sample = self.watch.get_sample()
                
                # Buffer samples
                if leap_sample is not None:
                    self.leap_buffer.append(leap_sample)
                if bioradio_sample is not None:
                    self.bioradio_buffer.append(bioradio_sample)
                if watch_sample is not None:
                    self.watch_buffer.append(watch_sample)
                
                # Update embodiment score at specified rate
                if current_time - last_update >= self.update_interval:
                    score = self._compute_embodiment_score()
                    
                    if score is not None:
                        self.score_history.append({
                            'timestamp': current_time,
                            'score': score
                        })
                        
                        # Display score
                        print(f"\rEmbodiment Score: {score:.1f}/100  ", end='')
                    
                    last_update = current_time
                
                # Small sleep to avoid busy waiting
                time.sleep(0.001)
        
        except KeyboardInterrupt:
            print("\n\nStopping monitoring...")
        
        finally:
            self.stop()
    
    def stop(self):
        """Stop monitoring"""
        self.is_running = False
        
        self.leap.stop_recording()
        self.bioradio.stop_recording()
        self.watch.stop_recording()
        
        print("\n✓ Monitoring stopped")
    
    def _compute_embodiment_score(self) -> Optional[float]:
        """
        Compute embodiment score from buffered data
        
        Returns:
            Embodiment score (0-100) or None if insufficient data
        """
        # Check if we have enough data
        if len(self.leap_buffer) < 10 or len(self.bioradio_buffer) < 10 or len(self.watch_buffer) < 10:
            return None
        
        try:
            # Convert buffers to arrays
            leap_data = self._buffer_to_dict(self.leap_buffer, 'leap')
            bioradio_data = self._buffer_to_dict(self.bioradio_buffer, 'bioradio')
            watch_data = self._buffer_to_dict(self.watch_buffer, 'watch')
            
            # Extract features
            features_dict = extract_all_features(leap_data, bioradio_data, watch_data)
            
            # Convert to array (must match training feature order)
            if self.model.feature_names is not None:
                feature_array = np.array([features_dict.get(name, 0.0) for name in self.model.feature_names])
            else:
                feature_array = np.array(list(features_dict.values()))
            
            # Predict
            score = self.model.predict(feature_array.reshape(1, -1))[0]
            
            return float(score)
        
        except Exception as e:
            print(f"\n⚠ Error computing score: {e}")
            return None
    
    def _buffer_to_dict(self, buffer: deque, sensor_type: str) -> dict:
        """Convert buffer to data dict"""
        if sensor_type == 'leap':
            return {
                'timestamps': np.array([s['timestamp'] for s in buffer]),
                'hand_position': np.array([s['hand_position'] for s in buffer]),
                'hand_velocity': np.array([s['hand_velocity'] for s in buffer]),
            }
        
        elif sensor_type == 'bioradio':
            return {
                'timestamps': np.array([s['timestamp'] for s in buffer]),
                'emg': np.array([s['emg'] for s in buffer]),
                'ecg': np.array([s['ecg'] for s in buffer]),
                'eda': np.array([s['eda'] for s in buffer]),
                'sampling_rate': 1000
            }
        
        elif sensor_type == 'watch':
            return {
                'timestamps': np.array([s['timestamp'] for s in buffer]),
                'heart_rate': np.array([s['heart_rate'] for s in buffer]),
                'accelerometer': np.array([s['accelerometer'] for s in buffer]) if 'accelerometer' in buffer[0] else None,
                'sampling_rate': 50
            }
    
    def get_current_score(self) -> Optional[float]:
        """Get most recent embodiment score"""
        if self.score_history:
            return self.score_history[-1]['score']
        return None
    
    def get_average_score(self, duration: float = 10.0) -> Optional[float]:
        """
        Get average score over recent period
        
        Args:
            duration: Time window (seconds)
        
        Returns:
            Average score or None
        """
        if not self.score_history:
            return None
        
        current_time = time.time()
        recent_scores = [
            s['score'] for s in self.score_history
            if current_time - s['timestamp'] <= duration
        ]
        
        if recent_scores:
            return np.mean(recent_scores)
        return None