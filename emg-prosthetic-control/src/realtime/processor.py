"""
RealtimeProcessor - Sensor-Based EMG/IMU Processor with Sliding Window & Filtering

Per-sensor EMG sampling rates:
    Avanti  sensors → 962.963  Hz  (1 EMG channel per sensor)
    Galileo sensors → 1259.2593 Hz (4 EMG channels per sensor)
"""

from __future__ import annotations
from collections import deque, defaultdict
from threading import Lock
from pathlib import Path
from typing import Dict, Optional, Tuple
import warnings
import numpy as np

import sys
src_path = Path(__file__).parent.parent
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from features.emg_features import extract_emg_features
from features.imu_features import extract_imu_features
from src.models.model_utils import load_model_bundle
from data import filtering


# ── Delsys sensor-type sampling rates ─────────────────────────────────────────
AVANTI_EMG_FS   = 962.963
GALILEO_EMG_FS  = 1259.2593
IMU_FS_DEFAULT  = 148.1481

# Galileo sensors have 4 EMG channels; Avanti have 1
GALILEO_EMG_CHANNELS = 4


# ── Channel map builder ────────────────────────────────────────────────────────

def build_channel_maps_from_delsys(delsys_connection):
    channel_info = delsys_connection.channel_info

    emg_by_sensor = defaultdict(list)
    imu_by_sensor = defaultdict(dict)
    guid_to_name  = {}

    IMU_EXPECTED_ORDER = ["ACC-X", "ACC-Y", "ACC-Z", "GYRO-X", "GYRO-Y", "GYRO-Z"]

    for guid, info in channel_info.items():
        name       = info["name"]
        chan_type  = info["type"]
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

    # Sort EMG channels per sensor; keep only 1- or 4-channel sensors
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


# ── RealtimeProcessor ──────────────────────────────────────────────────────────

class RealtimeProcessor:
    """Real-time EMG/IMU processor with per-sensor-type sampling rates."""

    def __init__(
        self,
        delsys_client,
        model_path: str,
        fs_imu: float = IMU_FS_DEFAULT,
        window_sec: float = 0.2,
        overlap_sec: float = 0.1,
        class_names: Optional[Dict[int, str]] = None,
        emg_filter: bool = True,
        imu_filter: bool = True,
        debug: bool = False
    ):
        self.debug      = debug
        self.fs_imu     = fs_imu
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        self.emg_filter = emg_filter
        self.imu_filter = imu_filter

        if self.debug:
            print("\nInitializing RealtimeProcessor...")

        # Load model bundle
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model bundle not found: {model_path}")
        self.pipeline_bundle   = load_model_bundle(model_path)
        self.pipeline          = self.pipeline_bundle.pipeline
        self.clf               = self.pipeline
        self.expected_features = getattr(self.pipeline_bundle, "feature_count", None)

        self.class_names = class_names or {
            0: "Neutral", 1: "Pinching", 2: "Grasping", 3: "Zipping"
        }

        # Build sensor maps from hardware
        self.emg_sensor_map, self.imu_sensor_map, self.guid_to_name = \
            build_channel_maps_from_delsys(delsys_client)

        if not self.emg_sensor_map or not self.imu_sensor_map:
            raise RuntimeError("No EMG or IMU sensors detected")

        # Assign per-sensor EMG fs based on channel count
        # Galileo = 4 channels → 1259.2593 Hz
        # Avanti  = 1 channel  →  962.963  Hz
        self.emg_fs_map: Dict[int, float] = {}
        for idx, guids in self.emg_sensor_map.items():
            if len(guids) == GALILEO_EMG_CHANNELS:
                self.emg_fs_map[idx] = GALILEO_EMG_FS
            else:
                self.emg_fs_map[idx] = AVANTI_EMG_FS

        if self.debug:
            for idx, fs in self.emg_fs_map.items():
                sensor_type = "Galileo" if fs == GALILEO_EMG_FS else "Avanti"
                print(f"  Sensor {idx}: {sensor_type} @ {fs} Hz")

        # Shared IMU window
        self.imu_win_size = int(window_sec * fs_imu)
        self.imu_step     = int((window_sec - overlap_sec) * fs_imu)

        # Build channel ordering and buffers
        self._build_channel_order()
        self._build_emg_groups()
        self._build_imu_groups()
        self._initialize_buffers()
        self.lock = Lock()
        self.total_predictions = 0

        if self.debug:
            print(f"✓ Processor ready ({self.expected_features} features expected)\n")

    # ── Internal builders ───────────────────────────────────────────────────

    def _build_channel_order(self):
        self.emg_channel_order = [
            guid
            for idx in sorted(self.emg_sensor_map.keys())
            for guid in self.emg_sensor_map[idx]
        ]
        self.imu_channel_order = [
            guid
            for idx in sorted(self.imu_sensor_map.keys())
            for guid in self.imu_sensor_map[idx]
        ]

    def _get_sensor_index(self, uuid) -> int:
        name = self.guid_to_name.get(uuid, "")
        try:
            return int(name.split()[1])
        except Exception:
            return -1

    def _build_emg_groups(self):
        """Group UUIDs by sensor. Each group shares the same fs."""
        self.emg_groups = []
        idx = 0
        while idx < len(self.emg_channel_order):
            remaining = len(self.emg_channel_order) - idx
            if remaining >= GALILEO_EMG_CHANNELS:
                slice_ = self.emg_channel_order[idx:idx + GALILEO_EMG_CHANNELS]
                sensor_idxs = [self._get_sensor_index(u) for u in slice_]
                if len(set(sensor_idxs)) == 1:
                    self.emg_groups.append(slice_)
                    idx += GALILEO_EMG_CHANNELS
                    continue
            self.emg_groups.append([self.emg_channel_order[idx]])
            idx += 1

    def _build_imu_groups(self):
        self.imu_groups = []
        n_axes = 6
        for i in range(0, len(self.imu_channel_order), n_axes):
            group = self.imu_channel_order[i:i + n_axes]
            if len(group) == n_axes:
                self.imu_groups.append(group)

    def _initialize_buffers(self):
        """Size each EMG buffer according to that sensor's actual fs."""
        self.emg_buffers = {}
        for idx in sorted(self.emg_sensor_map.keys()):
            fs       = self.emg_fs_map[idx]
            win_size = int(self.window_sec * fs)
            for guid in self.emg_sensor_map[idx]:
                self.emg_buffers[guid] = deque(maxlen=win_size * 2)

        self.imu_buffers = {
            uuid: deque(maxlen=self.imu_win_size * 2)
            for uuid in self.imu_channel_order
        }

    # ── Data ingestion ──────────────────────────────────────────────────────

    def add_raw_data(self, packet: Dict[str, np.ndarray]):
        with self.lock:
            for uuid, data in packet.items():
                arr = np.atleast_1d(data)
                if uuid in self.emg_buffers:
                    self.emg_buffers[uuid].extend(arr)
                elif uuid in self.imu_buffers:
                    self.imu_buffers[uuid].extend(arr)

    # ── Window readiness ────────────────────────────────────────────────────

    def _emg_win_size_for(self, uuid) -> int:
        sensor_idx = self._get_sensor_index(uuid)
        fs = self.emg_fs_map.get(sensor_idx, AVANTI_EMG_FS)
        return int(self.window_sec * fs)

    def is_window_ready(self) -> bool:
        emg_ok = all(
            len(self.emg_buffers[uuid]) >= self._emg_win_size_for(uuid)
            for uuid in self.emg_channel_order
        )
        imu_ok = all(
            len(buf) >= self.imu_win_size
            for buf in self.imu_buffers.values()
        )
        return emg_ok and imu_ok

    # ── Filtering ────────────────────────────────────────────────────────────

    def _filter_emg(self, data: np.ndarray, fs: float) -> np.ndarray:
        return filtering.preprocess_emg(data, fs=fs) if self.emg_filter else data

    def _filter_imu(self, data: np.ndarray) -> np.ndarray:
        return filtering.preprocess_imu(data, fs=self.fs_imu, lowpass=True, cutoff=20.0) \
               if self.imu_filter else data

    # ── Feature extraction ───────────────────────────────────────────────────

    def extract_features(self) -> Optional[np.ndarray]:
        with self.lock:
            if not self.is_window_ready():
                return None

            features = []

            # EMG — each group uses its own fs
            for group in self.emg_groups:
                sensor_idx = self._get_sensor_index(group[0])
                fs         = self.emg_fs_map.get(sensor_idx, AVANTI_EMG_FS)
                win_size   = int(self.window_sec * fs)

                group_windows = [
                    self._filter_emg(
                        np.array(self.emg_buffers[uuid])[-win_size:],
                        fs=fs
                    )
                    for uuid in group
                ]
                # Average across channels if multi-channel (Galileo)
                # This matches training: extract_emg_features averages axis=1 internally
                group_window = (
                    np.mean(group_windows, axis=0)
                    if len(group_windows) > 1
                    else group_windows[0]
                )
                features.extend(extract_emg_features(group_window, fs=fs))

            # IMU — shared fs
            for group in self.imu_groups:
                imu_arrays = []
                for uuid in group:
                    buf = np.array(self.imu_buffers[uuid])
                    if len(buf) < self.imu_win_size:
                        padded = np.zeros(self.imu_win_size)
                        padded[-len(buf):] = buf
                        buf_window = padded
                    else:
                        buf_window = buf[-self.imu_win_size:]
                    imu_arrays.append(self._filter_imu(buf_window))

                imu_window = np.column_stack(imu_arrays)
                features.extend(extract_imu_features(imu_window))

            features_array = np.array(features).reshape(1, -1)

            if self.expected_features and features_array.shape[1] != self.expected_features:
                raise RuntimeError(
                    f"Feature mismatch: got {features_array.shape[1]}, "
                    f"expected {self.expected_features}"
                )

            if self.debug:
                print(
                    f"DEBUG: Feature vector stats | "
                    f"min: {features_array.min():.4f} "
                    f"max: {features_array.max():.4f} "
                    f"mean: {features_array.mean():.4f} "
                    f"sum: {features_array.sum():.4f}"
                )
                print(f"DEBUG: First 10 features: {features_array.flatten()[:10]}")

            self._slide_buffers()
            return features_array

    # ── Buffer sliding ───────────────────────────────────────────────────────

    def _slide_buffers(self):
        for uuid in self.emg_channel_order:
            sensor_idx = self._get_sensor_index(uuid)
            fs         = self.emg_fs_map.get(sensor_idx, AVANTI_EMG_FS)
            step       = int((self.window_sec - self.overlap_sec) * fs)
            buf        = self.emg_buffers[uuid]
            for _ in range(min(step, len(buf))):
                buf.popleft()

        imu_step = int((self.window_sec - self.overlap_sec) * self.fs_imu)
        for uuid in self.imu_buffers:
            buf = self.imu_buffers[uuid]
            for _ in range(min(imu_step, len(buf))):
                buf.popleft()

    # ── Prediction ───────────────────────────────────────────────────────────

    def predict(self) -> Optional[Tuple[int, np.ndarray, str]]:
        features = self.extract_features()
        if features is None:
            return None

        pred_class = self.pipeline.predict(features)[0]
        pred_probs = self.pipeline.predict_proba(features)[0]
        class_name = self.class_names.get(pred_class, f"Class {pred_class}")
        self.total_predictions += 1

        if self.debug:
            print(f"DEBUG: Raw predict_proba output: {pred_probs}")
            print(f"DEBUG: Max prob={pred_probs.max():.3f}, Sum={pred_probs.sum():.3f}")
            print(f"[{self.total_predictions}] {class_name} | "
                  f"Confidence: {pred_probs.max() * 100:.1f}%")

        return pred_class, pred_probs, class_name

    # ── Utilities ────────────────────────────────────────────────────────────

    def reset_buffers(self):
        with self.lock:
            for buf in self.emg_buffers.values():
                buf.clear()
            for buf in self.imu_buffers.values():
                buf.clear()

    def get_buffer_status(self) -> dict:
        emg_fills = {
            uuid: len(self.emg_buffers[uuid])
            for uuid in self.emg_channel_order
        }
        imu_fills = {
            uuid: len(self.imu_buffers[uuid])
            for uuid in self.imu_channel_order
        }
        return {
            "emg_buffer_fill": emg_fills,
            "imu_buffer_fill": imu_fills,
            "window_ready":    self.is_window_ready(),
        }