"""
RealtimeProcessor - Sensor-Based EMG/IMU Processor with Sliding Window + Debug

Features:
- Sensor-based EMG mapping (supports 1 or 4 EMG per sensor)
- IMU axis mapping in ACC-X,Y,Z / GYRO-X,Y,Z order
- Sliding-window feature extraction with configurable overlap
- Loads model from model_bundle.pkl (no separate scaler)
- Thread-safe buffers
- Detailed debug statements for real-time diagnosis
"""

from __future__ import annotations
from collections import deque, defaultdict
from threading import Lock
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
import numpy as np
import sys

# Project imports
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from features.emg_features import extract_emg_features
from features.imu_features import extract_imu_features
from src.models.model_utils import load_model_bundle


def build_channel_maps_from_delsys(delsys_connection):
    """Build EMG & IMU sensor maps from Delsys channel_info"""
    channel_info = delsys_connection.channel_info

    emg_by_sensor = defaultdict(list)
    imu_by_sensor = defaultdict(dict)
    guid_to_name = {}

    IMU_EXPECTED_ORDER = ["ACC-X", "ACC-Y", "ACC-Z", "GYRO-X", "GYRO-Y", "GYRO-Z"]

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

    # Sort EMG channels
    emg_sensor_map = {}
    for idx, ch_list in emg_by_sensor.items():
        sorted_uuids = [guid for _, guid in sorted(ch_list)]
        if len(sorted_uuids) in (1, 4):
            emg_sensor_map[idx] = sorted_uuids

    # Order IMU channels
    imu_sensor_map = {}
    for idx, axis_dict in imu_by_sensor.items():
        ordered = [axis_dict.get(axis) for axis in IMU_EXPECTED_ORDER]
        if all(uuid is not None for uuid in ordered):
            imu_sensor_map[idx] = ordered

    print(f"✓ Found {len(emg_sensor_map)} EMG sensors, {len(imu_sensor_map)} IMU sensors")
    return emg_sensor_map, imu_sensor_map, guid_to_name


class RealtimeProcessor:
    """Real-time EMG/IMU processor with sliding-window feature extraction & classification + debug."""

    def __init__(
        self,
        delsys_client,
        model_path: str,
        fs_emg: float = 963.0,
        fs_imu: float = 148.148,
        window_sec: float = 0.206,  # ~198 EMG samples
        overlap_sec: float = 0.1,
        class_names: Optional[Dict[int, str]] = None
    ):
        print(f"\nInitializing RealtimeProcessor...")

        self.fs_emg = fs_emg
        self.fs_imu = fs_imu
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec

        # Sliding window sizes
        self.emg_win_size = int(window_sec * fs_emg)
        self.imu_win_size = int(window_sec * fs_imu)
        self.emg_step = int((window_sec - overlap_sec) * fs_emg)
        self.imu_step = int((window_sec - overlap_sec) * fs_imu)

        # Load model bundle
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model bundle not found: {model_path}")
        self.pipeline_bundle = load_model_bundle(model_path)
        self.clf = self.pipeline_bundle.pipeline
        self.scaler = getattr(self.pipeline_bundle, "scaler", None)
        self.expected_features = getattr(self.pipeline_bundle, "feature_count", None)

        if self.scaler is None:
            warnings.warn("No scaler found in model bundle; skipping feature scaling.")
            self.validate_features = False
        else:
            self.validate_features = True

        self.class_names = class_names or {0: "Neutral", 1: "Pinching", 2: "Grasping", 3: "Zipping"}

        # Sensor maps
        self.emg_sensor_map, self.imu_sensor_map, self.guid_to_name = build_channel_maps_from_delsys(delsys_client)
        if not self.emg_sensor_map or not self.imu_sensor_map:
            raise RuntimeError("No EMG or IMU sensors detected")

        # Build channel order & buffers
        self._build_channel_order()
        self._initialize_buffers()
        self.lock = Lock()
        self._build_emg_groups()
        self._build_imu_groups()
        self.total_predictions = 0

        print(f"✓ Processor ready ({self.expected_features} features expected)\n")
        print(f"  EMG sensors: {len(self.emg_sensor_map)} ({sum(len(v) for v in self.emg_sensor_map.values())} channels)")
        print(f"  IMU sensors: {len(self.imu_sensor_map)} ({sum(len(v) for v in self.imu_sensor_map.values())} channels)")

    def _build_channel_order(self):
        self.emg_channel_order = [guid for idx in sorted(self.emg_sensor_map.keys())
                                  for guid in self.emg_sensor_map[idx]]
        self.imu_channel_order = [guid for idx in sorted(self.imu_sensor_map.keys())
                                  for guid in self.imu_sensor_map[idx]]

    def _get_sensor_index(self, uuid):
        name = self.guid_to_name.get(uuid, "")
        try:
            return int(name.split()[1])
        except:
            return -1

    def _build_emg_groups(self):
        self.emg_groups = []
        idx = 0
        while idx < len(self.emg_channel_order):
            remaining = len(self.emg_channel_order) - idx
            if remaining >= 4:
                sensor_idxs = [self._get_sensor_index(uuid) for uuid in self.emg_channel_order[idx:idx+4]]
                if len(set(sensor_idxs)) == 1:
                    self.emg_groups.append(self.emg_channel_order[idx:idx+4])
                    idx += 4
                    continue
            self.emg_groups.append([self.emg_channel_order[idx]])
            idx += 1

    def _build_imu_groups(self):
        self.imu_groups = []
        n_axes = 6
        for i in range(0, len(self.imu_channel_order), n_axes):
            group = self.imu_channel_order[i:i+n_axes]
            if len(group) == n_axes:
                self.imu_groups.append(group)

    def _initialize_buffers(self):
        self.emg_buffers = {uuid: deque(maxlen=self.emg_win_size*2) for uuid in self.emg_channel_order}
        self.imu_buffers = {uuid: deque(maxlen=self.imu_win_size*2) for uuid in self.imu_channel_order}

    def add_raw_data(self, packet: Dict[str, np.ndarray]):
        with self.lock:
            for uuid, data in packet.items():
                arr = np.atleast_1d(data)
                if uuid in self.emg_buffers:
                    self.emg_buffers[uuid].extend(arr)
                elif uuid in self.imu_buffers:
                    self.imu_buffers[uuid].extend(arr)

    def is_window_ready(self) -> bool:
        return all(len(buf) >= self.emg_win_size for buf in self.emg_buffers.values()) and \
               all(len(buf) >= self.imu_win_size for buf in self.imu_buffers.values())

    def extract_features(self) -> Optional[np.ndarray]:
        with self.lock:
            if not self.is_window_ready():
                return None

            features = []

            # EMG features
            emg_feats_total = []
            for group in self.emg_groups:
                group_windows = np.array([np.array(self.emg_buffers[uuid])[-self.emg_win_size:] for uuid in group])
                window = np.mean(group_windows, axis=0) if group_windows.shape[0] > 1 else group_windows.flatten()
                feats = extract_emg_features(window, fs=self.fs_emg)
                emg_feats_total.extend(feats)
            features.extend(emg_feats_total)

            # IMU features
            imu_feats_total = []
            for group in self.imu_groups:
                imu_arrays = []
                for uuid in group:
                    buf = np.array(self.imu_buffers[uuid])
                    if len(buf) >= self.imu_win_size:
                        imu_arrays.append(buf[-self.imu_win_size:])
                    else:
                        padded = np.zeros(self.imu_win_size)
                        padded[-len(buf):] = buf
                        imu_arrays.append(padded)
                imu_window = np.column_stack(imu_arrays)
                feats = extract_imu_features(imu_window)
                imu_feats_total.extend(feats)
            features.extend(imu_feats_total)

            # --- DEBUG: inspect features before scaling ---
            features_array = np.array(features).reshape(1, -1)
            print("DEBUG: Feature vector stats | min:", features_array.min(),
                  "max:", features_array.max(),
                  "mean:", features_array.mean(),
                  "sum:", features_array.sum())
            print("DEBUG: First 10 features:", features_array[0, :10])

            # Validate
            if self.validate_features and self.expected_features and features_array.shape[1] != self.expected_features:
                raise RuntimeError(f"Feature mismatch: got {features_array.shape[1]}, expected {self.expected_features}")

            # Scale
            features_scaled = features_array
            if self.scaler is not None:
                features_scaled = self.scaler.transform(features_array)

            # --- DEBUG: after scaling ---
            print("DEBUG: Scaled features | min:", features_scaled.min(),
                  "max:", features_scaled.max(),
                  "mean:", features_scaled.mean(),
                  "sum:", features_scaled.sum())

            # Slide buffers
            self._slide_buffers()
            return features_scaled

    def _slide_buffers(self):
        for uuid in self.emg_buffers:
            buf = self.emg_buffers[uuid]
            for _ in range(min(self.emg_step, len(buf))):
                buf.popleft()
        for uuid in self.imu_buffers:
            buf = self.imu_buffers[uuid]
            for _ in range(min(self.imu_step, len(buf))):
                buf.popleft()

    def predict(self) -> Optional[Tuple[int, np.ndarray, str]]:
        features = self.extract_features()
        if features is None:
            return None

        # Ensure the pipeline is used
        pred_class = self.pipeline.predict(features)[0]
        pred_probs = self.pipeline.predict_proba(features)[0]

        class_name = self.class_names.get(pred_class, f"Class {pred_class}")

        # Debug
        print(f"DEBUG: Prediction -> class: {pred_class}, probs: {pred_probs}, name: {class_name}")
        print(f"DEBUG: Max prob={pred_probs.max():.3f}, Sum={pred_probs.sum():.3f}")

        self.total_predictions += 1
        return pred_class, pred_probs, class_name

    def reset_buffers(self):
        with self.lock:
            for buf in self.emg_buffers.values():
                buf.clear()
            for buf in self.imu_buffers.values():
                buf.clear()
            print("✓ Buffers reset")