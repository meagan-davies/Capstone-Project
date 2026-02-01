"""
Real-Time Data Processor - Fixed for Actual Delsys Channel Format

The actual Delsys channels are named just "EMG 1", "ACC X", etc.
WITHOUT sensor numbers in the name.

We need to use the sensor_index from channel_info to group them.
"""

import numpy as np
from collections import deque, defaultdict
from threading import Lock
from typing import Dict, List, Optional
import sys
from pathlib import Path

# Import your feature extraction functions
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from features.emg_features import extract_emg_features
from features.imu_features import extract_imu_features


def build_channel_maps_from_delsys(delsys_connection):
    """
    Build channel maps using sensor_index from channel_info.
    
    The Delsys channels are named "EMG 1", "ACC X", etc. without sensor numbers.
    We need to use the 'sensor_index' field to group them by sensor.
    
    Args:
        delsys_connection: DelsysClient with .trigno_base.channels and .channel_info
    
    Returns:
        emg_map: {sensor_idx: [uuid1, uuid2, uuid3, uuid4]}
        imu_map: {sensor_idx: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]}
        guid_to_name: {uuid: full_name} for debugging
    """
    # Access channel_info directly from the client
    try:
        channel_info = delsys_connection.channel_info
    except AttributeError:
        print("❌ DelsysClient doesn't have channel_info attribute")
        return {}, {}, {}
    
    emg_by_sensor = defaultdict(list)
    imu_by_sensor = defaultdict(dict)
    guid_to_name = {}
    
    for guid, info in channel_info.items():
        name = info['name']
        chan_type = info['type']
        sensor_idx = info['sensor_index']
        
        guid_to_name[guid] = f"Sensor {sensor_idx} | {name}"
        
        # EMG channels
        if chan_type == 'EMG':
            try:
                ch_num = int(name.split()[-1])
                emg_by_sensor[sensor_idx].append((ch_num, guid))
            except (ValueError, IndexError):
                pass
        
        # IMU channels
        elif chan_type == 'ACC':
            try:
                axis = name.split()[-1].upper()
                imu_by_sensor[sensor_idx][f"ACC-{axis}"] = guid
            except (ValueError, IndexError):
                pass
        
        elif chan_type == 'GYRO':
            try:
                axis = name.split()[-1].upper()
                imu_by_sensor[sensor_idx][f"GYRO-{axis}"] = guid
            except (ValueError, IndexError):
                pass
    
    # Sort and validate EMG sensors
    emg_final = {}
    for sensor_idx, channels_list in emg_by_sensor.items():
        sorted_channels = [guid for _, guid in sorted(channels_list)]
        if len(sorted_channels) in [1, 4]:
            emg_final[sensor_idx] = sorted_channels
    
    # Validate IMU sensors
    imu_final = {}
    for sensor_idx, channels_dict in imu_by_sensor.items():
        expected = ['ACC-X', 'ACC-Y', 'ACC-Z', 'GYRO-X', 'GYRO-Y', 'GYRO-Z']
        channel_list = [channels_dict.get(key) for key in expected]
        
        if all(ch is not None for ch in channel_list):
            imu_final[sensor_idx] = channel_list
    
    print(f"✓ Found {len(emg_final)} EMG sensors, {len(imu_final)} IMU sensors")
    
    return emg_final, imu_final, guid_to_name


class RealtimeProcessor:
    """
    Processes streaming EMG and IMU data in real-time.
    Works with actual Delsys channel format (uses sensor_index from channel_info).
    """
    
    def __init__(
        self,
        delsys_client,
        fs_emg: float = 963,
        fs_imu: float = 148.148,
        window_sec: float = 0.20,
        overlap_sec: float = 0.10
    ):
        """
        Initialize processor with DelsysClient.
        
        Args:
            delsys_client: Your DelsysClient instance (NOT trigno_base)
            fs_emg: EMG sampling frequency (Hz)
            fs_imu: IMU sampling frequency (Hz)
            window_sec: Window size in seconds
            overlap_sec: Overlap between windows in seconds
        """
        print(f"\nInitializing RealtimeProcessor...")
        
        self.fs_emg = fs_emg
        self.fs_imu = fs_imu
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        
        # Window sizes
        self.emg_win_size = int(window_sec * fs_emg)
        self.imu_win_size = int(window_sec * fs_imu)
        self.emg_step = int((window_sec - overlap_sec) * fs_emg)
        self.imu_step = int((window_sec - overlap_sec) * fs_imu)
        
        # Build channel maps using sensor_index
        self.emg_sensor_map, self.imu_sensor_map, self.guid_to_name = \
            build_channel_maps_from_delsys(delsys_client)
        
        if not self.emg_sensor_map or not self.imu_sensor_map:
            print(f"⚠️  WARNING: Some sensors missing!")
        
        # Create reverse lookup: UUID -> sensor_idx (for EMG)
        self.uuid_to_emg_sensor = {}
        for sensor_idx, uuid_list in self.emg_sensor_map.items():
            for uuid in uuid_list:
                self.uuid_to_emg_sensor[uuid] = sensor_idx
        
        # Buffers
        self.emg_buffers = {
            sensor_idx: deque(maxlen=self.emg_win_size * 2)
            for sensor_idx in self.emg_sensor_map.keys()
        }
        
        self.imu_buffers = {}
        for sensor_idx, uuid_list in self.imu_sensor_map.items():
            for uuid in uuid_list:
                self.imu_buffers[uuid] = deque(maxlen=self.imu_win_size * 2)
        
        # Sample counters
        self.emg_sample_count = 0
        self.imu_sample_count = 0
        
        # Thread safety
        self.lock = Lock()
        
        print(f"✓ Processor ready\n")
    
    def add_raw_data(self, raw_data: Dict[str, np.ndarray]):
        """
        Add streaming data from Delsys.
        
        Args:
            raw_data: {uuid: numpy_array} from Delsys stream
        """
        with self.lock:
            # Organize EMG data by sensor
            emg_data_by_sensor = defaultdict(list)
            
            for uuid, data in raw_data.items():
                # Check if this is an EMG channel
                if uuid in self.uuid_to_emg_sensor:
                    sensor_idx = self.uuid_to_emg_sensor[uuid]
                    emg_data_by_sensor[sensor_idx].append(data)
                
                # Check if this is an IMU channel
                elif uuid in self.imu_buffers:
                    self.imu_buffers[uuid].extend(data)
                    self.imu_sample_count += len(data)
            
            # Stack EMG channels per sensor and add to buffer
            for sensor_idx, channel_data_list in emg_data_by_sensor.items():
                if len(channel_data_list) == 0:
                    continue
                
                # Stack channels horizontally
                if len(channel_data_list) == 1:
                    # Single channel - reshape to (n_samples, 1)
                    stacked = channel_data_list[0].reshape(-1, 1)
                else:
                    # Multiple channels - stack normally
                    stacked = np.column_stack(channel_data_list)
                
                # Add each row to the deque
                for row in stacked:
                    self.emg_buffers[sensor_idx].append(row)
                
                self.emg_sample_count += stacked.shape[0]
    
    def is_window_ready(self) -> bool:
        """Check if we have enough data for feature extraction"""
        if not self.emg_buffers or not self.imu_buffers:
            return False
        
        emg_ready = all(len(buf) >= self.emg_win_size for buf in self.emg_buffers.values())
        imu_ready = all(len(buf) >= self.imu_win_size for buf in self.imu_buffers.values())
        emg_step_ready = self.emg_sample_count >= self.emg_step
        imu_step_ready = self.imu_sample_count >= self.imu_step
        
        return emg_ready and imu_ready and emg_step_ready and imu_step_ready
    
    def extract_window_features(self) -> Optional[np.ndarray]:
        """Extract features from current window"""
        with self.lock:
            if not self.is_window_ready():
                return None
            
            # Extract windows
            emg_windows = {
                idx: np.array(list(buf)[-self.emg_win_size:])
                for idx, buf in self.emg_buffers.items()
            }
            
            imu_windows = {
                uuid: np.array(list(buf)[-self.imu_win_size:])
                for uuid, buf in self.imu_buffers.items()
            }
            
            # Reset step counters
            self.emg_sample_count = 0
            self.imu_sample_count = 0
        
        features = []
        
        # EMG features per sensor (sorted by sensor index for consistency)
        for sensor_idx in sorted(self.emg_sensor_map.keys()):
            sensor_emg = emg_windows[sensor_idx]
            
            if sensor_emg.shape[1] == 1:
                # Single channel (Avanti)
                channel_data = sensor_emg[:, 0]
                emg_feats = extract_emg_features(channel_data, fs=self.fs_emg)
            else:
                # Multi-channel (Galileo)
                emg_feats = extract_emg_features(sensor_emg, fs=self.fs_emg)
            
            features.extend(emg_feats)
        
        # IMU features per sensor (sorted by sensor index for consistency)
        for sensor_idx in sorted(self.imu_sensor_map.keys()):
            uuid_list = self.imu_sensor_map[sensor_idx]
            sensor_data = np.column_stack([imu_windows[uuid] for uuid in uuid_list])
            imu_feats = extract_imu_features(sensor_data)
            features.extend(imu_feats)
        
        return np.array(features)
    
    def get_buffer_status(self) -> Dict:
        """Get current buffer status for debugging"""
        with self.lock:
            return {
                'emg_buffer_sizes': {idx: len(buf) for idx, buf in self.emg_buffers.items()},
                'imu_buffer_sizes': {uuid[:8]+'...': len(buf) for uuid, buf in self.imu_buffers.items()},
                'emg_sample_count': self.emg_sample_count,
                'imu_sample_count': self.imu_sample_count,
                'window_ready': self.is_window_ready()
            }
    
    def print_channel_mapping(self):
        """Print the channel UUID to name mapping for debugging"""
        print("\n" + "="*70)
        print("CHANNEL MAPPING")
        print("="*70)
        
        print("\nEMG Channels:")
        for sensor_idx, uuid_list in sorted(self.emg_sensor_map.items()):
            print(f"\n  Sensor {sensor_idx}:")
            for i, uuid in enumerate(uuid_list, 1):
                name = self.guid_to_name.get(uuid, 'Unknown')
                print(f"    Ch {i}: {uuid[:8]}... → {name}")
        
        print("\nIMU Channels:")
        for sensor_idx, uuid_list in sorted(self.imu_sensor_map.items()):
            print(f"\n  Sensor {sensor_idx}:")
            for uuid in uuid_list:
                name = self.guid_to_name.get(uuid, 'Unknown')
                print(f"    {uuid[:8]}... → {name}")
        
        print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    print("""
TO USE THIS PROCESSOR:

In realtime_classify.py, change line 117 from:
    processor = RealtimeProcessor(client.trigno_base)
    
To:
    processor = RealtimeProcessor(client)
    
That's it! The processor now expects the DelsysClient directly,
not client.trigno_base.
    """)