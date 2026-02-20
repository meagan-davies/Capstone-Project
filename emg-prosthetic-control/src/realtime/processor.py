"""
Real-Time Data Processor - Improved Version with Sliding Window Buffering

Key updates:
1. Ensures full windows (e.g., 198 EMG samples) are accumulated before feature extraction.
2. Sliding window applied per channel with configurable overlap.
3. Supports IMU buffering similarly.
4. Maintains thread safety and feature validation.
"""

import joblib
import numpy as np
from collections import deque, defaultdict
from threading import Lock
from typing import Dict, Optional, Tuple
import sys
from pathlib import Path
import warnings

# Import feature extraction functions
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
        imu_sensor_map: dict[int, list[uuid]]
        guid_to_name: dict[uuid, str]
    """
    try:
        channel_info = delsys_connection.channel_info
    except AttributeError:
        raise RuntimeError("DelsysClient doesn't have channel_info attribute")

    emg_by_sensor = defaultdict(list)
    imu_by_sensor = defaultdict(dict)
    guid_to_name = {}

    IMU_EXPECTED_ORDER = [
        "ACC-X", "ACC-Y", "ACC-Z",
        "GYRO-X", "GYRO-Y", "GYRO-Z"
    ]

    for guid, info in channel_info.items():
        name = info["name"]
        chan_type = info["type"]
        sensor_idx = info["sensor_index"]

        if sensor_idx is None:
            print(f"Warning: sensor_idx is None: {info}")
            continue

        guid_to_name[guid] = f"Sensor {sensor_idx} | {name}"

        if chan_type == "EMG":
            # TODO: Write a conditional for any specific cases where this fails.
            ch_num = int(name.split()[-1])
            emg_by_sensor[sensor_idx].append((ch_num, guid))

        elif chan_type in {"ACC", "GYRO"}:
            # TODO: Write a conditional for any specific cases where this fails.
            axis = name.split()[-1].upper()
            imu_by_sensor[sensor_idx][f"{chan_type}-{axis}"] = guid
        
        else:
            print(f"Ignoring channel chan_type={chan_type}: {info}")
            
    emg_sensor_map = {}
    for sensor_idx, ch_list in emg_by_sensor.items():
        sorted_uuids = [guid for _, guid in sorted(ch_list)]
        if len(sorted_uuids) in (1, 4):
            emg_sensor_map[sensor_idx] = sorted_uuids

    imu_sensor_map = {}
    for sensor_idx, axis_dict in imu_by_sensor.items():
        ordered = [axis_dict.get(axis) for axis in IMU_EXPECTED_ORDER]
        if all(uuid is not None for uuid in ordered):
            imu_sensor_map[sensor_idx] = ordered

    print(f"✓ Found {len(emg_sensor_map)} EMG sensors, {len(imu_sensor_map)} IMU sensors")
    return emg_sensor_map, imu_sensor_map, guid_to_name


class RealtimeProcessor:
    """Real-time EMG/IMU processor with sliding-window feature extraction and classification."""

    def __init__(
        self,
        delsys_client,
        model_path: str,
        scaler_path: str,
        fs_emg: float = 963,
        fs_imu: float = 148.148,
        window_sec: float = 0.206,  # ~198 samples for EMG
        overlap_sec: float = 0.1,
        expected_features: Optional[int] = None,
        validate_features: bool = True,
        class_names: Optional[Dict[int, str]] = None
    ):
        print(f"\nInitializing RealtimeProcessor...")

        self.fs_emg = fs_emg
        self.fs_imu = fs_imu
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        self.validate_features = validate_features

        # Compute sample window sizes
        self.emg_win_size = int(window_sec * fs_emg)
        self.imu_win_size = int(window_sec * fs_imu)
        self.emg_step = int((window_sec - overlap_sec) * fs_emg)
        self.imu_step = int((window_sec - overlap_sec) * fs_imu)

        # Load model & scaler
        try:
            self.clf = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Loaded model: {type(self.clf).__name__}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model/scaler: {e}")

        if expected_features is None:
            if hasattr(self.scaler, 'n_features_in_'):
                expected_features = self.scaler.n_features_in_
            else:
                warnings.warn("Could not auto-detect feature count. Disabling validation.")
                self.validate_features = False
        self.expected_features = expected_features

        self.class_names = class_names or {0: "Neutral", 1: "Pinching", 2: "Grasping", 3: "Zipping"}

        # Build channel maps
        self.emg_sensor_map, self.imu_sensor_map, self.guid_to_name = build_channel_maps_from_delsys(delsys_client)
        if not self.emg_sensor_map or not self.imu_sensor_map:
            raise RuntimeError("No EMG or IMU sensors detected")

        self._build_channel_order()
        self._initialize_buffers()
        self.lock = Lock()

        self._build_emg_groups()
        self._build_imu_groups()
        self.total_predictions = 0
        print(f"✓ Processor ready ({self.expected_features} features expected)\n")

    def _build_channel_order(self):
        # TODO: Figure out what this is doing and confirm it's right.
        self.emg_channel_order = [guid for idx in sorted(self.emg_sensor_map.keys()) for guid in self.emg_sensor_map[idx]]
        self.imu_channel_order = [guid for idx in sorted(self.imu_sensor_map.keys()) for guid in self.imu_sensor_map[idx]]

    def _get_sensor_index(self, uuid):
        # FIXME: Do not use try-except here.
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
        """Buffers now ensure full windows before extraction."""
        self.emg_buffers = {uuid: deque(maxlen=self.emg_win_size * 2) for uuid in self.emg_channel_order}
        self.imu_buffers = {uuid: deque(maxlen=self.imu_win_size * 2) for uuid in self.imu_channel_order}

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
            # EMG must have full window
            if not all(len(buf) >= self.emg_win_size for buf in self.emg_buffers.values()):
                print("DEBUG: EMG window not full yet")
                return None

            features = []

            # -----------------------
            # EMG FEATURES
            # -----------------------
            emg_feats_total = []
            for group in self.emg_groups:
                group_windows = np.array([np.array(self.emg_buffers[uuid])[-self.emg_win_size:] for uuid in group])
                window = np.mean(group_windows, axis=0) if group_windows.shape[0] > 1 else group_windows.flatten()
                feats = extract_emg_features(window, fs=self.fs_emg)
                emg_feats_total.extend(feats)
            features.extend(emg_feats_total)

            # Debug EMG
            print(f"DEBUG: EMG buffers (last 5 samples per channel):")
            for uuid in self.emg_channel_order:
                print(f"  {self.guid_to_name[uuid]}: {list(self.emg_buffers[uuid])[-5:]}")
            print(f"DEBUG: Extracted EMG features ({len(emg_feats_total)}): {emg_feats_total[:10]} ...")

            # -----------------------
            # IMU FEATURES
            # -----------------------
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

            # Debug IMU
            print(f"DEBUG: IMU buffers (last 5 samples per axis):")
            for uuid in self.imu_channel_order:
                print(f"  {self.guid_to_name[uuid]}: {list(self.imu_buffers[uuid])[-5:]}")
            print(f"DEBUG: Extracted IMU features ({len(imu_feats_total)}): {imu_feats_total[:10]} ...")

            # -----------------------
            # Convert, validate, scale
            # -----------------------
            features_array = np.array(features).reshape(1, -1)
            print(f"DEBUG: Total feature vector shape: {features_array.shape}")
            if self.validate_features and features_array.shape[1] != self.expected_features:
                raise RuntimeError(f"Feature mismatch: got {features_array.shape[1]}, expected {self.expected_features}")
            
            features_scaled = self.scaler.transform(features_array)
            print(f"DEBUG: Scaled features (first 10): {features_scaled[0, :10]} ...")

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
        pred_class = self.clf.predict(features)[0]
        pred_probs = self.clf.predict_proba(features)[0]
        class_name = self.class_names.get(pred_class, f"Class {pred_class}")

        # Debug prediction
        print(f"DEBUG: Prediction -> class: {pred_class}, probs: {pred_probs}, name: {class_name}")

        self.total_predictions += 1
        return pred_class, pred_probs, class_name


    def get_buffer_status(self) -> Dict:
        with self.lock:
            return {
                'emg_buffer_fill': min(len(buf) for buf in self.emg_buffers.values()),
                'imu_buffer_fill': min(len(buf) for buf in self.imu_buffers.values()),
                'window_ready': self.is_window_ready(),
                'total_predictions': self.total_predictions
            }

    def reset_buffers(self):
        with self.lock:
            for buf in self.emg_buffers.values():
                buf.clear()
            for buf in self.imu_buffers.values():
                buf.clear()
            print("✓ Buffers reset")

    def print_window_sizes(self):
        with self.lock:
            print("=== Current buffer sizes ===")
            print("EMG channels:")
            for uuid in self.emg_channel_order:
                print(f"  {self.guid_to_name[uuid]}: {len(self.emg_buffers[uuid])} samples")
            print("IMU channels:")
            for uuid in self.imu_channel_order:
                print(f"  {self.guid_to_name[uuid]}: {len(self.imu_buffers[uuid])} samples")
            print("============================")
        