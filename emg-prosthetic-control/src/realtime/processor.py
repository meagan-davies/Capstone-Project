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

    Returns:
        emg_sensor_map: dict[int, list[uuid]]
            sensor_idx -> ordered EMG channel UUIDs
        imu_sensor_map: dict[int, list[uuid]]
            sensor_idx -> ordered IMU UUIDs:
            [ACC-X, ACC-Y, ACC-Z, GYRO-X, GYRO-Y, GYRO-Z]
        guid_to_name: dict[uuid, str]
            UUID -> human-readable name
    """
    try:
        channel_info = delsys_connection.channel_info
    except AttributeError:
        print("❌ DelsysClient doesn't have channel_info attribute")
        return {}, {}, {}

    emg_by_sensor = defaultdict(list)
    imu_by_sensor = defaultdict(dict)
    guid_to_name = {}

    IMU_EXPECTED_ORDER = [
        "ACC-X", "ACC-Y", "ACC-Z",
        "GYRO-X", "GYRO-Y", "GYRO-Z"
    ]

    # ----------------------------
    # Parse raw channel info
    # ----------------------------
    for guid, info in channel_info.items():
        name = info.get("name", "")
        chan_type = info.get("type", "")
        sensor_idx = info.get("sensor_index", None)

        if sensor_idx is None:
            continue

        guid_to_name[guid] = f"Sensor {sensor_idx} | {name}"

        # ---------- EMG ----------
        if chan_type == "EMG":
            # Try to extract EMG channel number for ordering
            try:
                ch_num = int(name.split()[-1])
            except (ValueError, IndexError):
                ch_num = 0

            emg_by_sensor[sensor_idx].append((ch_num, guid))

        # ---------- IMU ----------
        elif chan_type in {"ACC", "GYRO"}:
            try:
                axis = name.split()[-1].upper()
                imu_by_sensor[sensor_idx][f"{chan_type}-{axis}"] = guid
            except (ValueError, IndexError):
                pass

    # ----------------------------
    # Finalize EMG map
    # ----------------------------
    emg_sensor_map = {}
    for sensor_idx, ch_list in emg_by_sensor.items():
        # Sort by channel number
        sorted_uuids = [guid for _, guid in sorted(ch_list)]

        # Accept 1-channel or 4-channel EMG sensors
        if len(sorted_uuids) in (1, 4):
            emg_sensor_map[sensor_idx] = sorted_uuids

    # ----------------------------
    # Finalize IMU map (ORDERED LIST)
    # ----------------------------
    imu_sensor_map = {}
    for sensor_idx, axis_dict in imu_by_sensor.items():
        ordered = [axis_dict.get(axis) for axis in IMU_EXPECTED_ORDER]

        # Only accept complete IMUs
        if all(uuid is not None for uuid in ordered):
            imu_sensor_map[sensor_idx] = ordered

    print(f"✓ Found {len(emg_sensor_map)} EMG sensors, {len(imu_sensor_map)} IMU sensors")

    return emg_sensor_map, imu_sensor_map, guid_to_name

class RealtimeProcessor:
    def __init__(self, delsys_client, fs_emg=963, fs_imu=148.148, 
                 window_sec=0.20, overlap_sec=0.10,
                 model_path="lda_model.pkl", scaler_path="lda_scaler.pkl"):
        
        print(f"\nInitializing RealtimeProcessor with model/scaler...")

        self.EXPECTED_FEATURES = 231

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
        
        if not self.emg_sensor_map:
            raise RuntimeError("No EMG sensors detected")

        if not self.imu_sensor_map:
            raise RuntimeError("No IMU sensors detected")

        # Reverse lookup: UUID -> sensor_idx
        self.uuid_to_emg_sensor = {}
        for sensor_idx, uuid_list in self.emg_sensor_map.items():
            for uuid in uuid_list:
                self.uuid_to_emg_sensor[uuid] = sensor_idx

        # Buffers
        self.emg_channel_order = []
        for sensor_idx in sorted(self.emg_sensor_map.keys()):
            self.emg_channel_order.extend(self.emg_sensor_map[sensor_idx])

        self.emg_buffers = {
            uuid: deque(maxlen=self.emg_win_size * 2)
            for uuid in self.emg_channel_order
        }

        self.imu_channel_order = []
        for sensor_idx in sorted(self.imu_sensor_map.keys()):
            self.imu_channel_order.extend(self.imu_sensor_map[sensor_idx])
            
        self.imu_buffers = {
            uuid: deque(maxlen=self.imu_win_size * 2)
            for uuid in self.imu_channel_order
        }

        self.emg_sample_count = 0
        self.imu_sample_count = 0
        self.lock = Lock()

        # Load trained model + scaler
        self.clf = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)

        print("EMG order:", self.emg_channel_order)
        print("IMU order:", self.imu_channel_order)

        print(f"✓ Processor ready with classifier '{type(self.clf).__name__}'\n")

    def add_raw_data(self, packet: dict):
        """Add raw Delsys packet to buffers."""
        with self.lock:
            for uuid, raw in packet.items():
                data = np.atleast_1d(raw)

                # EMG
                if uuid in self.emg_buffers:
                    self.emg_buffers[uuid].extend(data)
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
            for uuid in self.emg_channel_order:
                buf = self.emg_buffers[uuid]
                window = np.array(buf)[-self.emg_win_size:]

                emg_feats = extract_emg_features(window, fs=self.fs_emg)
                features.extend(emg_feats)

            # IMU
            imu_data = []
            for uuid in self.imu_channel_order:
                buf = self.imu_buffers[uuid]
                imu_data.append(np.array(buf)[-self.imu_win_size:])

            sensor_data = np.column_stack(imu_data)
            imu_feats = extract_imu_features(sensor_data)
            features.extend(imu_feats)

            features_array = np.array(features).reshape(1, -1)

            if features_array.shape[1] != self.EXPECTED_FEATURES:

                raise RuntimeError(
                    f"Feature count mismatch: got {features_array.shape[1]}, "
                    f"expected {self.EXPECTED_FEATURES}"
                )

            features_scaled = self.scaler.transform(features_array)

            # Slide buffers after extraction
            for uuid in self.emg_channel_order:
                buf = self.emg_buffers[uuid]
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
