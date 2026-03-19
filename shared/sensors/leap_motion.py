"""
Leap Motion sensor interface
Used by both prosthetic control (ground truth) and embodiment model (control features)
"""

import numpy as np
from typing import Dict, Optional, Tuple
import time


class LeapMotionCapture:
    """
    Interface for Leap Motion hand tracking
    
    Provides:
    - Hand position tracking
    - Velocity and acceleration
    - Gesture detection
    - Tracking quality metrics
    """
    
    def __init__(self, sampling_rate: int = 115):
        """
        Initialize Leap Motion capture
        
        Args:
            sampling_rate: Target sampling rate (Hz)
        """
        self.sampling_rate = sampling_rate
        self.is_recording = False
        self.data_buffer = {
            'timestamps': [],
            'hand_position': [],
            'hand_velocity': [],
            'tracking_confidence': [],
            'hand_visible': []
        }
        
        # TODO: learn more about this connection process doesn't seem in line with leapmotion docs
        # Try to import Leap SDK
        try:
            import Leap
            self.controller = Leap.Controller()
            self.sdk_available = True
            print("✓ Leap Motion SDK connected")
        except ImportError:
            print("⚠ Leap Motion SDK not found - using simulation mode")
            self.sdk_available = False
            self.controller = None
    
    def start_recording(self) -> None:
        """Start recording Leap Motion data"""
        self.is_recording = True
        self.data_buffer = {
            'timestamps': [],
            'hand_position': [],
            'hand_velocity': [],
            'tracking_confidence': [],
            'hand_visible': []
        }
        print("Started Leap Motion recording")
    
    def get_frame(self) -> Optional[Dict]:
        """
        Get current frame data
        
        Returns:
            Dict with hand tracking data or None if no hand detected
        """
        if not self.sdk_available:
            return self._get_simulated_frame()
        
        frame = self.controller.frame()
        
        if not frame.hands.is_empty:
            hand = frame.hands.frontmost
            
            frame_data = {
                'timestamp': time.time(),
                'hand_position': np.array([
                    hand.palm_position.x,
                    hand.palm_position.y,
                    hand.palm_position.z
                ]),
                'hand_velocity': np.array([
                    hand.palm_velocity.x,
                    hand.palm_velocity.y,
                    hand.palm_velocity.z
                ]),
                'tracking_confidence': hand.confidence,
                'hand_visible': True
            }
            
            if self.is_recording:
                self.data_buffer['timestamps'].append(frame_data['timestamp'])
                self.data_buffer['hand_position'].append(frame_data['hand_position'])
                self.data_buffer['hand_velocity'].append(frame_data['hand_velocity'])
                self.data_buffer['tracking_confidence'].append(frame_data['tracking_confidence'])
                self.data_buffer['hand_visible'].append(frame_data['hand_visible'])
            
            return frame_data
        else:
            # No hand detected
            if self.is_recording:
                self.data_buffer['timestamps'].append(time.time())
                self.data_buffer['hand_position'].append(np.array([np.nan, np.nan, np.nan]))
                self.data_buffer['hand_velocity'].append(np.array([np.nan, np.nan, np.nan]))
                self.data_buffer['tracking_confidence'].append(0.0)
                self.data_buffer['hand_visible'].append(False)
            
            return None
    
    def stop_recording(self) -> Dict:
        """
        Stop recording and return collected data
        
        Returns:
            Dict with numpy arrays of recorded data
        """
        self.is_recording = False
        
        # Convert lists to numpy arrays
        recorded_data = {
            'timestamps': np.array(self.data_buffer['timestamps']),
            'hand_position': np.array(self.data_buffer['hand_position']),
            'hand_velocity': np.array(self.data_buffer['hand_velocity']),
            'tracking_confidence': np.array(self.data_buffer['tracking_confidence']),
            'hand_visible': np.array(self.data_buffer['hand_visible']),
            'sampling_rate': self.sampling_rate
        }
        
        print(f"Stopped recording. Captured {len(recorded_data['timestamps'])} frames")
        return recorded_data
    
    def _get_simulated_frame(self) -> Dict:
        """Generate simulated Leap Motion data for testing"""
        t = time.time()
        
        # Simulate smooth hand movement
        x = 10 * np.sin(2 * np.pi * 0.5 * t)
        y = 200 + 5 * np.cos(2 * np.pi * 0.3 * t)
        z = -100 + 3 * np.sin(2 * np.pi * 0.4 * t)
        
        frame_data = {
            'timestamp': t,
            'hand_position': np.array([x, y, z]),
            'hand_velocity': np.array([
                10 * np.pi * np.cos(2 * np.pi * 0.5 * t),
                -5 * 0.6 * np.pi * np.sin(2 * np.pi * 0.3 * t),
                3 * 0.8 * np.pi * np.cos(2 * np.pi * 0.4 * t)
            ]),
            'tracking_confidence': 0.95,
            'hand_visible': True
        }
        
        if self.is_recording:
            self.data_buffer['timestamps'].append(frame_data['timestamp'])
            self.data_buffer['hand_position'].append(frame_data['hand_position'])
            self.data_buffer['hand_velocity'].append(frame_data['hand_velocity'])
            self.data_buffer['tracking_confidence'].append(frame_data['tracking_confidence'])
            self.data_buffer['hand_visible'].append(frame_data['hand_visible'])
        
        return frame_data
    
    def get_tracking_quality(self, data: Dict) -> float:
        """
        Calculate tracking quality metric
        
        Args:
            data: Recorded data dict
        
        Returns:
            Quality score (0-1)
        """
        visible_frames = np.sum(data['hand_visible'])
        total_frames = len(data['hand_visible'])
        
        if total_frames == 0:
            return 0.0
        
        visibility_ratio = visible_frames / total_frames
        avg_confidence = np.mean(data['tracking_confidence'][data['hand_visible']])
        
        quality = 0.7 * visibility_ratio + 0.3 * avg_confidence
        return quality


def calculate_tracking_metrics(leap_data: Dict, target_trajectory: Optional[np.ndarray] = None) -> Dict:
    """
    Calculate tracking performance metrics
    
    Args:
        leap_data: Data from LeapMotionCapture
        target_trajectory: Optional target positions (n_samples, 3)
    
    Returns:
        Dict of tracking metrics
    """
    from ..control_metrics import (
        calculate_tracking_error,
        calculate_movement_smoothness,
        calculate_path_efficiency
    )
    
    hand_pos = leap_data['hand_position']
    
    # Remove NaN values (when hand not visible)
    valid_mask = ~np.isnan(hand_pos).any(axis=1)
    hand_pos_clean = hand_pos[valid_mask]
    
    metrics = {
        'tracking_reliability': np.mean(valid_mask),
        'movement_smoothness': calculate_movement_smoothness(hand_pos_clean),
    }
    
    if target_trajectory is not None and len(target_trajectory) == len(hand_pos_clean):
        metrics['tracking_error'] = calculate_tracking_error(hand_pos_clean, target_trajectory)
        metrics['path_efficiency'] = calculate_path_efficiency(
            hand_pos_clean,
            target_trajectory[0],
            target_trajectory[-1]
        )
    
    return metrics