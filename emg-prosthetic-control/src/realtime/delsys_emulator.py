"""
Delsys Hardware Emulator

Simulates the Delsys Trigno system for testing without real hardware.
Implements the same interface as DelsysClient so you can swap them easily.

Usage:
    # Instead of:
    from src.realtime.delsys_client import DelsysClient
    client = DelsysClient(key, license)
    
    # Use:
    from src.realtime.delsys_emulator import DelsysEmulator
    client = DelsysEmulator()
"""

import numpy as np
import time
from typing import Dict, List, Optional
from collections import deque
import uuid


class DelsysEmulator:
    """
    Emulates Delsys Trigno hardware for testing without real sensors.
    
    Provides the same interface as DelsysClient but generates synthetic
    EMG and IMU data instead of connecting to real hardware.
    """
    
    def __init__(self, 
                 n_sensors: int = 4,
                 noise_level: float = 0.1,
                 signal_freq: float = 50.0):
        """
        Initialize emulator.
        
        Args:
            n_sensors: Number of sensors to simulate (default: 4)
            noise_level: Amount of noise to add (0-1, default: 0.1)
            signal_freq: Frequency of simulated EMG signal in Hz (default: 50)
        """
        self.n_sensors = n_sensors
        self.noise_level = noise_level
        self.signal_freq = signal_freq
        
        # EMG sampling rate (from Delsys specs)
        self.fs_emg = 963  # Hz
        self.fs_imu = 148.148  # Hz
        
        # State
        self.is_connected = False
        self.is_configured = False
        self.is_streaming = False
        
        # Sensor and channel info
        self.sensors = []
        self.channel_guids = []
        self.channel_info = {}
        
        # Data generation
        self.time_counter = 0
        self.packet_count = 0
        self.data_buffer = deque(maxlen=1000)
        
        # Gesture simulation (changes every 5 seconds)
        self.current_gesture = 0
        self.gesture_change_time = time.time()
        self.gesture_duration = 5.0  # seconds
        
        print("✓ Delsys Emulator initialized")
        print(f"  Sensors: {n_sensors}")
        print(f"  Noise level: {noise_level}")
    
    def connect(self) -> bool:
        """Simulate connection to base station"""
        print("Emulator: Connecting to virtual base station...")
        time.sleep(0.5)  # Simulate connection delay
        
        self.is_connected = True
        print("✓ Emulator: Connected to virtual base station")
        print("  Pipeline state: Connected")
        return True
    
    def scan_sensors(self) -> bool:
        """Simulate sensor scanning"""
        print("Emulator: Scanning for virtual sensors...")
        time.sleep(1.0)  # Simulate scan delay
        
        # Create simulated sensors
        self.sensors = []
        for i in range(self.n_sensors):
            sensor = {
                'pair_number': i + 1,
                'sid': 10000 + i,
                'mode': 'Default (EMG + IMU)',
                'name': f'Virtual Sensor {i+1}'
            }
            self.sensors.append(sensor)
        
        print(f"✓ Emulator: Found {len(self.sensors)} virtual sensor(s)")
        for i, sensor in enumerate(self.sensors):
            print(f"  Sensor {i+1}: Pair#{sensor['pair_number']}, "
                  f"SID:{sensor['sid']}, Mode:{sensor['mode']}")
        
        return True
    
    def configure(self) -> bool:
        """Simulate configuration"""
        print("Emulator: Configuring virtual data collection...")
        
        # Create channel GUIDs and info
        self.channel_guids = []
        self.channel_info = {}
        
        for sensor_idx in range(self.n_sensors):
            # EMG channel
            emg_guid = str(uuid.uuid4())
            self.channel_guids.append(emg_guid)
            self.channel_info[emg_guid] = {
                'name': f'EMG {sensor_idx + 1}',
                'type': 'EMG',
                'sample_rate': self.fs_emg,
                'unit': 'mV',
                'sensor_index': sensor_idx
            }
            
            # IMU channels (3 ACC + 3 GYRO = 6 per sensor)
            for axis in ['X', 'Y', 'Z']:
                # Accelerometer
                acc_guid = str(uuid.uuid4())
                self.channel_guids.append(acc_guid)
                self.channel_info[acc_guid] = {
                    'name': f'ACC {axis} {sensor_idx + 1}',
                    'type': f'ACC',
                    'sample_rate': self.fs_imu,
                    'unit': 'g',
                    'sensor_index': sensor_idx
                }
                
                # Gyroscope
                gyro_guid = str(uuid.uuid4())
                self.channel_guids.append(gyro_guid)
                self.channel_info[gyro_guid] = {
                    'name': f'GYRO {axis} {sensor_idx + 1}',
                    'type': f'GYRO',
                    'sample_rate': self.fs_imu,
                    'unit': 'deg/s',
                    'sensor_index': sensor_idx
                }
        
        emg_count = sum(1 for c in self.channel_info.values() if c['type'] == 'EMG')
        acc_count = sum(1 for c in self.channel_info.values() if c['type'] == 'ACC')
        gyro_count = sum(1 for c in self.channel_info.values() if c['type'] == 'GYRO')
        
        print(f"✓ Emulator: Configured {len(self.channel_guids)} virtual channels")
        print(f"  Channel breakdown:")
        print(f"    EMG: {emg_count}")
        print(f"    ACC: {acc_count}")
        print(f"    GYRO: {gyro_count}")
        print("  Pipeline state: Armed")
        
        self.is_configured = True
        return True
    
    def start_streaming(self) -> bool:
        """Start simulated data streaming"""
        print("Emulator: Starting virtual data stream...")
        
        self.is_streaming = True
        self.time_counter = 0
        self.packet_count = 0
        
        print("✓ Emulator: Virtual data streaming started")
        print("  Pipeline state: Running")
        return True
    
    def _generate_emg_signal(self, n_samples: int, gesture: int) -> np.ndarray:
        """
        Generate synthetic EMG signal.
        
        Different gestures have different signal characteristics:
        - Gesture 0 (Neutral): Low amplitude, mostly noise
        - Gesture 1 (Pinching): Medium amplitude, 50 Hz primary
        - Gesture 2 (Grasping): High amplitude, 50 Hz primary
        - Gesture 3 (Zipping): Medium amplitude, 30 Hz primary, modulated
        """
        t = (np.arange(n_samples) + self.time_counter) / self.fs_emg
        
        if gesture == 0:  # Neutral
            signal = 0.05 * np.random.randn(n_samples)
        
        elif gesture == 1:  # Pinching
            signal = (
                0.3 * np.sin(2 * np.pi * self.signal_freq * t) +
                0.1 * np.sin(2 * np.pi * self.signal_freq * 2 * t) +
                self.noise_level * np.random.randn(n_samples)
            )
        
        elif gesture == 2:  # Grasping
            signal = (
                0.6 * np.sin(2 * np.pi * self.signal_freq * t) +
                0.2 * np.sin(2 * np.pi * self.signal_freq * 2 * t) +
                self.noise_level * np.random.randn(n_samples)
            )
        
        else:  # Zipping
            modulation = 0.5 * (1 + np.sin(2 * np.pi * 2 * t))
            signal = (
                0.4 * modulation * np.sin(2 * np.pi * 30 * t) +
                self.noise_level * np.random.randn(n_samples)
            )
        
        return signal
    
    def _generate_imu_signal(self, n_samples: int, gesture: int, axis: str) -> np.ndarray:
        """
        Generate synthetic IMU signal (ACC or GYRO).
        
        Different gestures have different motion patterns.
        """
        t = (np.arange(n_samples) + self.time_counter) / self.fs_imu
        
        if gesture == 0:  # Neutral - minimal movement
            signal = 0.01 * np.random.randn(n_samples)
            if axis == 'ACC':
                signal += 1.0  # Gravity component
        
        elif gesture == 1:  # Pinching - small precise movements
            signal = (
                0.2 * np.sin(2 * np.pi * 3 * t) +
                0.05 * np.random.randn(n_samples)
            )
            if axis == 'ACC':
                signal += 1.0
        
        elif gesture == 2:  # Grasping - larger movements
            signal = (
                0.5 * np.sin(2 * np.pi * 2 * t) +
                0.1 * np.random.randn(n_samples)
            )
            if axis == 'ACC':
                signal += 1.0
        
        else:  # Zipping - repetitive motion
            signal = (
                0.3 * np.sin(2 * np.pi * 4 * t) +
                0.05 * np.random.randn(n_samples)
            )
            if axis == 'ACC':
                signal += 1.0
        
        return signal
    
    def poll_data(self) -> Optional[Dict[str, np.ndarray]]:
        """
        Generate and return synthetic data packet.
        
        Returns data in the same format as real Delsys hardware:
        Dictionary mapping channel GUIDs (strings) to data arrays
        """
        if not self.is_streaming:
            return None
        
        # Check if we should change gesture (every 5 seconds)
        current_time = time.time()
        if current_time - self.gesture_change_time > self.gesture_duration:
            self.current_gesture = (self.current_gesture + 1) % 4
            self.gesture_change_time = current_time
            gesture_names = ['Neutral', 'Pinching', 'Grasping', 'Zipping']
            print(f"\nEmulator: Gesture changed to {gesture_names[self.current_gesture]}")
        
        # Generate data for each channel
        data_dict = {}
        
        for guid, info in self.channel_info.items():
            if info['type'] == 'EMG':
                # EMG: Generate ~10 samples (simulating ~100 Hz poll rate)
                n_samples = 10
                signal = self._generate_emg_signal(n_samples, self.current_gesture)
                data_dict[guid] = signal
            
            elif info['type'] in ['ACC', 'GYRO']:
                # IMU: Generate ~2 samples
                n_samples = 2
                axis_char = info['name'].split()[1]  # Extract X, Y, or Z
                signal = self._generate_imu_signal(
                    n_samples, 
                    self.current_gesture,
                    info['type']
                )
                data_dict[guid] = signal
        
        self.time_counter += 10  # Advance time
        self.packet_count += 1
        
        return data_dict
    
    def stop_streaming(self):
        """Stop simulated streaming"""
        if self.is_streaming:
            self.is_streaming = False
            print("✓ Emulator: Stopped virtual streaming")
            print("  Pipeline state: Armed")
    
    def reset_pipeline(self):
        """Reset/disarm pipeline"""
        if self.is_configured:
            self.is_configured = False
            print("✓ Emulator: Pipeline reset")
            print("  Pipeline state: Connected")
    
    def disconnect(self):
        """Disconnect from emulator"""
        if self.is_streaming:
            self.stop_streaming()
        if self.is_configured:
            self.reset_pipeline()
        
        self.is_connected = False
        print("✓ Emulator: Disconnected")
    
    def get_emg_channel_guids(self) -> List[str]:
        """Get EMG channel GUIDs"""
        return [
            guid for guid, info in self.channel_info.items()
            if info['type'] == 'EMG'
        ]
    
    def get_imu_channel_guids(self) -> List[str]:
        """Get IMU channel GUIDs (ACC + GYRO)"""
        return [
            guid for guid, info in self.channel_info.items()
            if info['type'] in ['ACC', 'GYRO']
        ]
    
    def get_pipeline_state(self) -> str:
        """Get current pipeline state"""
        if not self.is_connected:
            return "Off"
        elif self.is_streaming:
            return "Running"
        elif self.is_configured:
            return "Armed"
        else:
            return "Connected"
    
    def get_total_packets(self) -> int:
        """Get total packets generated"""
        return self.packet_count


# Test the emulator
if __name__ == "__main__":
    print("="*70)
    print("TESTING DELSYS EMULATOR")
    print("="*70)
    
    # Create emulator
    emulator = DelsysEmulator(n_sensors=4, noise_level=0.1)
    
    # Test connection
    print("\n[TEST 1] Connection")
    assert emulator.connect()
    
    # Test scanning
    print("\n[TEST 2] Sensor Scan")
    assert emulator.scan_sensors()
    
    # Test configuration
    print("\n[TEST 3] Configuration")
    assert emulator.configure()
    
    # Test streaming
    print("\n[TEST 4] Data Streaming")
    assert emulator.start_streaming()
    
    print("\nGenerating data for 5 seconds...")
    start_time = time.time()
    packet_count = 0
    
    while time.time() - start_time < 5:
        data = emulator.poll_data()
        if data:
            packet_count += 1
            if packet_count % 10 == 0:
                print(f"\rPackets: {packet_count}, Channels: {len(data)}", 
                      end='', flush=True)
        time.sleep(0.01)
    
    print(f"\n✓ Received {packet_count} data packets")
    
    # Test cleanup
    print("\n[TEST 5] Cleanup")
    emulator.stop_streaming()
    emulator.disconnect()
    
    print("\n" + "="*70)
    print("✓ ALL TESTS PASSED")
    print("="*70)
    print("\nEmulator is working correctly!")
    print("You can now use it to test your real-time system without hardware.")