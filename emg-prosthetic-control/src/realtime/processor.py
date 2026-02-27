"""
Real-Time Data Processor
Updated to support:

- ModelBundle loading
- Pipeline-based inference (scaler inside)
- Optional EMG filtering
- Proper feature validation
"""

import numpy as np
from collections import deque, defaultdict
from threading import Lock
from typing import Dict, Optional, Tuple
import sys
from pathlib import Path
import warnings
from scipy.signal import butter, filtfilt

# Ensure src in path
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from features.emg_features import extract_emg_features
from features.imu_features import extract_imu_features
from models.model_utils import load_model_bundle


# Realtime Processor
class RealtimeProcessor:

    def __init__(
        self,
        delsys_client,
        model_name: str,
        fs_emg: float = 963,
        fs_imu: float = 148.148,
        window_sec: float = 0.206,
        overlap_sec: float = 0.1,
        validate_features: bool = True,
        use_emg_filter: bool = True,
        class_names: Optional[Dict[int, str]] = None
    ):

        print("\nInitializing RealtimeProcessor...")

        # Sampling + window config
        self.fs_emg = fs_emg
        self.fs_imu = fs_imu
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        self.validate_features = validate_features
        self.use_emg_filter = use_emg_filter

        self.emg_win_size = int(window_sec * fs_emg)
        self.imu_win_size = int(window_sec * fs_imu)
        self.emg_step = int((window_sec - overlap_sec) * fs_emg)
        self.imu_step = int((window_sec - overlap_sec) * fs_imu)

        # Load ModelBundle
        bundle = load_model_bundle(model_name)
        self.pipeline = bundle.pipeline
        self.expected_features = bundle.feature_count

        print(f"✓ Loaded model bundle: {model_name}")
        print(f"✓ Expected feature count: {self.expected_features}")

        # Optional EMG bandpass filter
        if self.use_emg_filter:
            self.b, self.a = butter(
                N=4,
                Wn=[20/(fs_emg/2), 450/(fs_emg/2)],
                btype="band"
            )
            print("✓ EMG bandpass filter enabled (20–450 Hz)")

        # Class names
        self.class_names = class_names or {
            0: "Neutral",
            1: "Pinching",
            2: "Grasping",
            3: "Zipping"
        }

        # Channel mapping
        self.emg_sensor_map, self.imu_sensor_map, self.guid_to_name = \
            self._build_channel_maps(delsys_client)

        self._build_channel_order()
        self._initialize_buffers()
        self.lock = Lock()

        self.total_predictions = 0

        print("✓ Processor ready\n")

    # Channel Mapping
    def _build_channel_maps(self, delsys_connection):

        channel_info = delsys_connection.channel_info

        emg_by_sensor = defaultdict(list)
        imu_by_sensor = defaultdict(dict)
        guid_to_name = {}

        IMU_ORDER = [
            "ACC-X", "ACC-Y", "ACC-Z",
            "GYRO-X", "GYRO-Y", "GYRO-Z"
        ]

        for guid, info in channel_info.items():

            name = info["name"]
            chan_type = info["type"]
            sensor_idx = info["sensor_index"]

            if sensor_idx is None:
                continue

            guid_to_name[guid] = f"Sensor {sensor_idx} | {name}"

            if chan_type == "EMG":
                ch_num = int(name.split()[-1])
                emg_by_sensor[sensor_idx].append((ch_num, guid))

            elif chan_type in {"ACC", "GYRO"}:
                axis = name.split()[-1].upper()
                imu_by_sensor[sensor_idx][f"{chan_type}-{axis}"] = guid

        emg_map = {
            idx: [guid for _, guid in sorted(ch)]
            for idx, ch in emg_by_sensor.items()
        }

        imu_map = {}
        for idx, axis_dict in imu_by_sensor.items():
            ordered = [axis_dict.get(axis) for axis in IMU_ORDER]
            if all(uuid is not None for uuid in ordered):
                imu_map[idx] = ordered

        print(f"✓ Found {len(emg_map)} EMG sensors, {len(imu_map)} IMU sensors")

        return emg_map, imu_map, guid_to_name

    # Buffer Handling
    def _build_channel_order(self):
        self.emg_channel_order = [
            guid for idx in sorted(self.emg_sensor_map.keys())
            for guid in self.emg_sensor_map[idx]
        ]

        self.imu_channel_order = [
            guid for idx in sorted(self.imu_sensor_map.keys())
            for guid in self.imu_sensor_map[idx]
        ]

    def _initialize_buffers(self):
        self.emg_buffers = {
            uuid: deque(maxlen=self.emg_win_size * 2)
            for uuid in self.emg_channel_order
        }

        self.imu_buffers = {
            uuid: deque(maxlen=self.imu_win_size * 2)
            for uuid in self.imu_channel_order
        }

    # Data Input
    def add_raw_data(self, packet: Dict[str, np.ndarray]):
        with self.lock:
            for uuid, data in packet.items():
                arr = np.atleast_1d(data)
                if uuid in self.emg_buffers:
                    self.emg_buffers[uuid].extend(arr)
                elif uuid in self.imu_buffers:
                    self.imu_buffers[uuid].extend(arr)

    def is_window_ready(self):
        return (
            all(len(buf) >= self.emg_win_size for buf in self.emg_buffers.values())
            and
            all(len(buf) >= self.imu_win_size for buf in self.imu_buffers.values())
        )

    # Feature Extraction
    def extract_features(self):

        with self.lock:

            if not self.is_window_ready():
                return None

            features = []

            # EMG
            for uuid in self.emg_channel_order:

                window = np.array(self.emg_buffers[uuid])[-self.emg_win_size:]

                if self.use_emg_filter:
                    window = filtfilt(self.b, self.a, window)

                feats = extract_emg_features(window, fs=self.fs_emg)
                features.extend(feats)

            # IMU
            for idx in sorted(self.imu_sensor_map.keys()):

                uuids = self.imu_sensor_map[idx]
                imu_arrays = [
                    np.array(self.imu_buffers[u])[-self.imu_win_size:]
                    for u in uuids
                ]

                imu_window = np.column_stack(imu_arrays)
                feats = extract_imu_features(imu_window)
                features.extend(feats)

            features_array = np.array(features).reshape(1, -1)

            if self.validate_features:
                if features_array.shape[1] != self.expected_features:
                    raise RuntimeError(
                        f"Feature mismatch: got {features_array.shape[1]}, "
                        f"expected {self.expected_features}"
                    )

            self._slide_buffers()

            return features_array

    # Sliding Window
    def _slide_buffers(self):
        for buf in self.emg_buffers.values():
            for _ in range(min(self.emg_step, len(buf))):
                buf.popleft()

        for buf in self.imu_buffers.values():
            for _ in range(min(self.imu_step, len(buf))):
                buf.popleft()

    # Prediction
    def predict(self) -> Optional[Tuple[int, np.ndarray, str]]:

        features = self.extract_features()
        if features is None:
            return None

        pred_class = self.pipeline.predict(features)[0]
        pred_probs = self.pipeline.predict_proba(features)[0]

        class_name = self.class_names.get(pred_class, f"Class {pred_class}")

        self.total_predictions += 1

        return pred_class, pred_probs, class_name