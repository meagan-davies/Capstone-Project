"""
Apple Watch sensor interface
Used by embodiment model for heart rate and motion data
"""

import numpy as np
from typing import Dict, Optional
import time


class AppleWatchCapture:
    """
    Interface for Apple Watch health and motion data
    
    Provides:
    - Heart rate (HR)
    - Heart rate variability (HRV)
    - Accelerometer
    - Gyroscope
    """
    
    def __init__(self, sampling_rate: int = 50):
        """
        Initialize Apple Watch capture
        
        Args:
            sampling_rate: Target sampling rate (Hz)
        """
        self.sampling_rate = sampling_rate
        self.is_recording = False
        self.data_buffer = {
            'timestamps': [],
            'heart_rate': [],
            'accelerometer': [],
            'gyroscope': []
        }
        
        # Try to connect to Apple Watch
        try:
            # TODO: Placeholder for HealthKit/WatchConnectivity integration
            # In practice, this would use iOS HealthKit API
            self.device_available = False
            print("⚠ Apple Watch connection not available - using simulation mode")
        except Exception as e:
            print(f"⚠ Apple Watch connection failed: {e}")
            self.device_available = False
    
    def start_recording(self) -> None:
        """Start recording Apple Watch data"""
        self.is_recording = True
        self.data_buffer = {
            'timestamps': [],
            'heart_rate': [],
            'accelerometer': [],
            'gyroscope': []
        }
        print("Started Apple Watch recording")
    
    def get_sample(self) -> Dict:
        """
        Get current sample from Apple Watch
        
        Returns:
            Dict with current sensor readings
        """
        if not self.device_available:
            return self._get_simulated_sample()
        
        # Actual device reading would go here
        pass
    
    def stop_recording(self) -> Dict:
        """
        Stop recording and return collected data
        
        Returns:
            Dict with numpy arrays of recorded data
        """
        self.is_recording = False
        
        recorded_data = {
            'timestamps': np.array(self.data_buffer['timestamps']),
            'heart_rate': np.array(self.data_buffer['heart_rate']),
            'accelerometer': np.array(self.data_buffer['accelerometer']),  # Shape: (n_samples, 3)
            'gyroscope': np.array(self.data_buffer['gyroscope']),  # Shape: (n_samples, 3)
            'sampling_rate': self.sampling_rate
        }
        
        print(f"Stopped recording. Captured {len(recorded_data['timestamps'])} samples")
        return recorded_data
    
    def _get_simulated_sample(self) -> Dict:
        """Generate simulated Apple Watch data for testing"""
        t = time.time()
        
        # Simulate heart rate (70-80 bpm with variability)
        base_hr = 75
        hr_variability = 5 * np.sin(2 * np.pi * 0.1 * t)
        heart_rate = base_hr + hr_variability + np.random.normal(0, 2)
        
        # Simulate wrist accelerometer (movement + gravity)
        accel_x = 9.81 * np.sin(2 * np.pi * 0.3 * t) + np.random.normal(0, 0.5)
        accel_y = 9.81 * np.cos(2 * np.pi * 0.3 * t) + np.random.normal(0, 0.5)
        accel_z = 9.81 + np.random.normal(0, 0.3)
        accelerometer = np.array([accel_x, accel_y, accel_z])
        
        # Simulate gyroscope (rotation rates)
        gyro_x = 0.1 * np.sin(2 * np.pi * 0.4 * t) + np.random.normal(0, 0.05)
        gyro_y = 0.1 * np.cos(2 * np.pi * 0.4 * t) + np.random.normal(0, 0.05)
        gyro_z = 0.05 * np.sin(2 * np.pi * 0.2 * t) + np.random.normal(0, 0.02)
        gyroscope = np.array([gyro_x, gyro_y, gyro_z])
        
        sample = {
            'timestamp': t,
            'heart_rate': heart_rate,
            'accelerometer': accelerometer,
            'gyroscope': gyroscope
        }
        
        if self.is_recording:
            self.data_buffer['timestamps'].append(sample['timestamp'])
            self.data_buffer['heart_rate'].append(sample['heart_rate'])
            self.data_buffer['accelerometer'].append(sample['accelerometer'])
            self.data_buffer['gyroscope'].append(sample['gyroscope'])
        
        return sample
    
    def calculate_hrv(self, duration: float = 60.0) -> Dict:
        """
        Calculate heart rate variability over a period
        
        Args:
            duration: Recording duration (seconds)
        
        Returns:
            Dict with HRV metrics
        """
        print(f"Calculating HRV over {duration}s...")
        
        self.start_recording()
        start_time = time.time()
        
        while time.time() - start_time < duration:
            self.get_sample()
            time.sleep(1.0 / self.sampling_rate)
        
        data = self.stop_recording()
        
        # Calculate HRV metrics (simplified)
        hr_values = data['heart_rate']
        rr_intervals = 60000 / hr_values  # Convert HR to RR intervals (ms)
        
        hrv_metrics = {
            'hr_mean': np.mean(hr_values),
            'hr_std': np.std(hr_values),
            'rmssd': np.sqrt(np.mean(np.diff(rr_intervals)**2)),
            'sdnn': np.std(rr_intervals)
        }
        
        print("✓ HRV calculated")
        return hrv_metrics