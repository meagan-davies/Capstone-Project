"""
BioRadio sensor interface
Used by embodiment model for EMG, ECG, and EDA signals
"""

import numpy as np
from typing import Dict, List, Optional
import time


class BioRadioCapture:
    """
    Interface for BioRadio biosignal acquisition
    
    Supports:
    - EMG (electromyography)
    - ECG (electrocardiography)
    - EDA/GSR (electrodermal activity)
    """
    
    def __init__(self, 
                 sampling_rate: int = 1000,
                 emg_channels: List[int] = [0, 1, 2, 3],
                 ecg_channel: int = 4,
                 eda_channel: int = 5):
        """
        Initialize BioRadio capture
        
        Args:
            sampling_rate: Sampling frequency (Hz)
            emg_channels: List of EMG channel indices
            ecg_channel: ECG channel index
            eda_channel: EDA/GSR channel index
        """
        self.sampling_rate = sampling_rate
        self.emg_channels = emg_channels
        self.ecg_channel = ecg_channel
        self.eda_channel = eda_channel
        
        self.is_recording = False
        self.data_buffer = {
            'timestamps': [],
            'emg': [],
            'ecg': [],
            'eda': []
        }
        
        # Try to connect to BioRadio device
        try:
            # TODO: Import BioRadio SDK (placeholder - replace with actual SDK)
            # import bioradio
            # self.device = bioradio.connect()
            self.device_available = False  # Change to True when SDK available
            print("⚠ BioRadio SDK not found - using simulation mode")
        except ImportError:
            print("⚠ BioRadio SDK not found - using simulation mode")
            self.device_available = False
    
    def start_recording(self) -> None:
        """Start recording BioRadio data"""
        self.is_recording = True
        self.data_buffer = {
            'timestamps': [],
            'emg': [],
            'ecg': [],
            'eda': []
        }
        print("Started BioRadio recording")
    
    def get_sample(self) -> Dict:
        """
        Get current sample from all channels
        
        Returns:
            Dict with current sensor readings
        """
        if not self.device_available:
            return self._get_simulated_sample()
        
        # Actual device reading would go here
        # sample = self.device.read()
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
            'emg': np.array(self.data_buffer['emg']),  # Shape: (n_samples, n_emg_channels)
            'ecg': np.array(self.data_buffer['ecg']),  # Shape: (n_samples,)
            'eda': np.array(self.data_buffer['eda']),  # Shape: (n_samples,)
            'sampling_rate': self.sampling_rate,
            'channel_info': {
                'emg_channels': self.emg_channels,
                'ecg_channel': self.ecg_channel,
                'eda_channel': self.eda_channel
            }
        }
        
        print(f"Stopped recording. Captured {len(recorded_data['timestamps'])} samples")
        return recorded_data
    
    def _get_simulated_sample(self) -> Dict:
        """Generate simulated BioRadio data for testing"""
        t = time.time()
        
        # Simulate EMG (muscle activity with noise)
        emg_sample = []
        for ch in self.emg_channels:
            baseline = 0.05 * np.sin(2 * np.pi * 0.5 * t + ch)
            noise = np.random.normal(0, 0.02)
            emg_sample.append(baseline + noise)
        
        # Simulate ECG (heartbeat pattern)
        heart_rate = 75  # bpm
        ecg_freq = heart_rate / 60
        ecg_sample = 1.0 * np.sin(2 * np.pi * ecg_freq * t) + 0.1 * np.random.normal()
        
        # Simulate EDA (slow varying arousal)
        eda_sample = 2.0 + 0.5 * np.sin(2 * np.pi * 0.1 * t) + 0.05 * np.random.normal()
        
        sample = {
            'timestamp': t,
            'emg': np.array(emg_sample),
            'ecg': ecg_sample,
            'eda': eda_sample
        }
        
        if self.is_recording:
            self.data_buffer['timestamps'].append(sample['timestamp'])
            self.data_buffer['emg'].append(sample['emg'])
            self.data_buffer['ecg'].append(sample['ecg'])
            self.data_buffer['eda'].append(sample['eda'])
        
        return sample
    
    def set_baseline(self, duration: float = 30.0) -> Dict:
        """
        Record baseline signals for normalization
        
        Args:
            duration: Baseline recording duration (seconds)
        
        Returns:
            Dict with baseline statistics
        """
        print(f"Recording {duration}s baseline...")
        
        self.start_recording()
        start_time = time.time()
        
        while time.time() - start_time < duration:
            self.get_sample()
            time.sleep(1.0 / self.sampling_rate)
        
        baseline_data = self.stop_recording()
        
        baseline_stats = {
            'emg_baseline': np.mean(np.abs(baseline_data['emg']), axis=0),
            'ecg_baseline': np.mean(baseline_data['ecg']),
            'eda_baseline': np.mean(baseline_data['eda']),
            'emg_std': np.std(baseline_data['emg'], axis=0),
            'ecg_std': np.std(baseline_data['ecg']),
            'eda_std': np.std(baseline_data['eda'])
        }
        
        print("✓ Baseline recorded")
        return baseline_stats