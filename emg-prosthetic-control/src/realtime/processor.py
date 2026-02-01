"""
Real-Time Data Processor - Fixed for Actual Delsys Channel Format

The actual Delsys channels are named just "EMG 1", "ACC X", etc.
WITHOUT sensor numbers in the name.

We need to use the sensor_index from channel_info to group them.
"""
import joblib
import numpy as np
from collections import deque, defaultdict
from threading import Lock
from typing import Dict, Optional
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
    Returns EMG and IMU sensor maps and guid-to-name mapping.
    """
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

        # EMG
        if chan_type == 'EMG':
            try:
                ch_num = int(name.split()[-1])
                emg_by_sensor[sensor_idx].append((ch_num, guid))
            except (ValueError, IndexError):
                emg_by_sensor[sensor_idx].append((0, guid))

        # IMU
        elif chan_type in ['ACC', 'GYRO']:
            try:
                axis = name.split()[-1].upper()
                imu_by_sensor[sensor_idx][f"{chan_type}-{axis}"] = guid
            except (ValueError, IndexError):
                pass

    # Sort EMG channels
    emg_final = {}
    for sensor_idx, channels_list in emg_by_sensor.items():
        sorted_channels = [guid for _, guid in sorted(channels_list)]
        if len(sorted_channels) in [1, 4]:
            emg_final[sensor_idx] = sorted_channels

    # Validate IMU sensors
    imu_final = {}
    expected = ['ACC-X','ACC-Y','ACC-Z','GYRO-X','GYRO-Y','GYRO-Z']
    for sensor_idx, ch_dict in imu_by_sensor.items():
        channel_list = [ch_dict.get(key) for key in expected]
        if all(ch is not None for ch in channel_list):
            imu_final[sensor_idx] = channel_list

    print(f"✓ Found {len(emg_final)} EMG sensors, {len(imu_final)} IMU sensors")
    return emg_final, imu_final, guid_to_name

class RealtimeProcessor:
    def __init__(self, delsys_client, fs_emg=963, fs_imu=148.148, 
                 window_sec=0.20, overlap_sec=0.10,
                 model_path="lda_model.pkl", scaler_path="lda_scaler.pkl"):
        print(f"\nInitializing RealtimeProcessor with model/scaler...")

        self.fs_emg = fs_emg
        self.fs_imu = fs_imu
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec

        # Window sizes
        self.emg_win_size = int(window_sec * fs_emg)
        self.imu_win_size = int(window_sec * fs_imu)
        self.emg_step = int((window_sec - overlap_sec) * fs_emg)
        self.imu_step = int((window_sec - overlap_sec) * fs_imu)

        # Build channel maps
        self.emg_sensor_map, self.imu_sensor_map, self.guid_to_name = \
            build_channel_maps_from_delsys(delsys_client)

        # Reverse lookup: UUID -> sensor_idx
        self.uuid_to_emg_sensor = {}
        for sensor_idx, uuid_list in self.emg_sensor_map.items():
            for uuid in uuid_list:
                self.uuid_to_emg_sensor[uuid] = sensor_idx

        # Buffers
        self.emg_buffers = {s: deque(maxlen=self.emg_win_size*2) for s in self.emg_sensor_map}
        self.imu_buffers = {uuid: deque(maxlen=self.imu_win_size*2)
                            for lst in self.imu_sensor_map.values() for uuid in lst}

        self.emg_sample_count = 0
        self.imu_sample_count = 0
        self.lock = Lock()

        # Load trained model + scaler
        self.clf = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        print(f"✓ Processor ready with classifier '{type(self.clf).__name__}'\n")

    def add_raw_data(self, packet: dict):
        """Add raw Delsys packet to buffers."""
        with self.lock:
            for uuid, raw in packet.items():
                data = np.atleast_1d(raw)

                # EMG
                if uuid in self.uuid_to_emg_sensor:
                    sensor_idx = self.uuid_to_emg_sensor[uuid]
                    self.emg_buffers[sensor_idx].extend(data)
                    self.emg_sample_count += len(data)

                # IMU (UUID-keyed ONLY)
                elif uuid in self.imu_buffers:
                    self.imu_buffers[uuid].extend(data)
                    self.imu_sample_count += len(data)

                # Unknown channel (ignore safely)
                else:
                    pass

    def is_window_ready(self) -> bool:
        # EMG
        for buf in self.emg_buffers.values():
            if len(buf) < self.emg_win_size:
                return False

        # IMU
        for buf in self.imu_buffers.values():
            if len(buf) < self.imu_win_size:
                return False

        return True

    def extract_window_features(self):
        """Extract features if window is ready, else return None."""
        with self.lock:
            if not self.is_window_ready():
                return None

            features = []

            # EMG
            for sensor_idx in sorted(self.emg_sensor_map.keys()):
                buf = self.emg_buffers[sensor_idx]
                window = np.array(list(buf)[-self.emg_win_size:])

                # Safe handling for 1D vs 2D arrays
                if window.ndim == 1:
                    emg_feats = extract_emg_features(window, fs=self.fs_emg)
                else:
                    emg_feats = extract_emg_features(window, fs=self.fs_emg)

                features.extend(emg_feats)

            # IMU
            imu_data = [
                np.array(buf)[-self.imu_win_size:]
                for buf in self.imu_buffers.values()
            ]

            sensor_data = np.column_stack(imu_data)
            imu_feats = extract_imu_features(sensor_data)
            features.extend(imu_feats)

            features_array = np.array(features).reshape(1, -1)
            features_scaled = self.scaler.transform(features_array)

            # Slide buffers after extraction
            for sensor_idx in self.emg_sensor_map:
                buf = self.emg_buffers[sensor_idx]
                for _ in range(min(self.emg_step, len(buf))):
                    buf.popleft()
                self.emg_sample_count = max(self.emg_sample_count - self.emg_step, 0)

            for buf in self.imu_buffers.values():
                for _ in range(min(self.imu_step, len(buf))):
                    buf.popleft()
                self.imu_sample_count = max(self.imu_sample_count - self.imu_step, 0)

        return features_scaled

    def predict_current_window(self):
        """Return predicted class and probabilities if window ready."""
        features_scaled = self.extract_window_features()
        if features_scaled is None:
            return None, None
        pred_class = self.clf.predict(features_scaled)[0]
        pred_probs = self.clf.predict_proba(features_scaled)[0]
        return pred_class, pred_probs

    def get_buffer_status(self):
        """Return current buffer status for EMG and IMU samples."""
        with self.lock:
            return {
                'emg_sample_count': self.emg_sample_count,
                'imu_sample_count': self.imu_sample_count
            }



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
