"""
Real-Time Data Processor - Improved Version

Key improvements:
1. Separate feature extraction from prediction
2. Better error handling and validation
3. Configurable feature count validation
4. More efficient buffer management
5. Better debugging capabilities
"""

import joblib
import numpy as np
from collections import deque, defaultdict
from threading import Lock
from typing import Dict, Optional, Tuple
import sys
from pathlib import Path
import warnings

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

    # Parse channels
    for guid, info in channel_info.items():
        name = info.get("name", "")
        chan_type = info.get("type", "")
        sensor_idx = info.get("sensor_index")

        if sensor_idx is None:
            continue

        guid_to_name[guid] = f"Sensor {sensor_idx} | {name}"

        # EMG channels
        if chan_type == "EMG":
            try:
                ch_num = int(name.split()[-1])
            except (ValueError, IndexError):
                ch_num = 0
            emg_by_sensor[sensor_idx].append((ch_num, guid))

        # IMU channels
        elif chan_type in {"ACC", "GYRO"}:
            try:
                axis = name.split()[-1].upper()
                imu_by_sensor[sensor_idx][f"{chan_type}-{axis}"] = guid
            except (ValueError, IndexError):
                pass

    # Finalize EMG map
    emg_sensor_map = {}
    for sensor_idx, ch_list in emg_by_sensor.items():
        sorted_uuids = [guid for _, guid in sorted(ch_list)]
        if len(sorted_uuids) in (1, 4):
            emg_sensor_map[sensor_idx] = sorted_uuids

    # Finalize IMU map
    imu_sensor_map = {}
    for sensor_idx, axis_dict in imu_by_sensor.items():
        ordered = [axis_dict.get(axis) for axis in IMU_EXPECTED_ORDER]
        if all(uuid is not None for uuid in ordered):
            imu_sensor_map[sensor_idx] = ordered

    print(f"✓ Found {len(emg_sensor_map)} EMG sensors, {len(imu_sensor_map)} IMU sensors")

    return emg_sensor_map, imu_sensor_map, guid_to_name

class RealtimeProcessor:
    """
    Real-time EMG/IMU data processor with integrated classification.
    
    Improvements:
    - Cleaner separation of concerns
    - Better error handling
    - More efficient buffer operations
    - Optional feature validation
    - Performance metrics tracking
    """
    
    def __init__(
        self,
        delsys_client,
        model_path: str,
        scaler_path: str,
        fs_emg: float = 963,
        fs_imu: float = 148.148,
        window_sec: float = 0.20,
        overlap_sec: float = 0.10,
        expected_features: Optional[int] = None,
        validate_features: bool = True,
        class_names: Optional[Dict[int, str]] = None
    ):
        """
        Initialize processor with classifier.
        
        Args:
            delsys_client: DelsysClient instance
            model_path: Path to trained model (.pkl)
            scaler_path: Path to feature scaler (.pkl)
            fs_emg: EMG sampling frequency (Hz)
            fs_imu: IMU sampling frequency (Hz)
            window_sec: Window duration (seconds)
            overlap_sec: Window overlap (seconds)
            expected_features: Expected feature count (None = auto-detect from model)
            validate_features: Whether to validate feature count
            class_names: Dict mapping class labels to names
        """
        print(f"\nInitializing RealtimeProcessor...")

        # Parameters
        self.fs_emg = fs_emg
        self.fs_imu = fs_imu
        self.window_sec = window_sec
        self.overlap_sec = overlap_sec
        self.validate_features = validate_features

        # Window sizes
        self.emg_win_size = int(window_sec * fs_emg)
        self.imu_win_size = int(window_sec * fs_imu)
        self.emg_step = int((window_sec - overlap_sec) * fs_emg)
        self.imu_step = int((window_sec - overlap_sec) * fs_imu)

        # Load model and scaler
        try:
            self.clf = joblib.load(model_path)
            self.scaler = joblib.load(scaler_path)
            print(f"✓ Loaded model: {type(self.clf).__name__}")
        except Exception as e:
            raise RuntimeError(f"Failed to load model/scaler: {e}")

        # Auto-detect expected features from scaler
        if expected_features is None:
            if hasattr(self.scaler, 'n_features_in_'):
                expected_features = self.scaler.n_features_in_
            else:
                warnings.warn("Could not auto-detect feature count. Disabling validation.")
                self.validate_features = False
        
        self.expected_features = expected_features
        
        # Class names
        self.class_names = class_names or {
            0: "Neutral",
            1: "Pinching",
            2: "Grasping",
            3: "Zipping"
        }

        # Build channel maps
        self.emg_sensor_map, self.imu_sensor_map, self.guid_to_name = \
            build_channel_maps_from_delsys(delsys_client)
        
        if not self.emg_sensor_map:
            raise RuntimeError("No EMG sensors detected")
        if not self.imu_sensor_map:
            raise RuntimeError("No IMU sensors detected")

        # Build ordered channel lists
        self._build_channel_order()
        
        # Initialize buffers
        self._initialize_buffers()
        
        # Counters
        self.emg_sample_count = 0
        self.imu_sample_count = 0
        self.prediction_count = 0
        
        # Thread safety
        self.lock = Lock()

        # Performance tracking
        self.total_predictions = 0
        self.feature_extraction_times = []

        self._build_emg_groups()
        self._build_imu_groups()

        print(f"✓ EMG groups: {len(self.emg_groups)} | IMU groups: {len(self.imu_groups)}")

        print(f"✓ Processor ready ({self.expected_features} features expected)\n")

    def _build_channel_order(self):
        """Build deterministic channel ordering for consistent feature extraction."""
        # EMG channels (sorted by sensor index, then channel number)
        self.emg_channel_order = []
        for sensor_idx in sorted(self.emg_sensor_map.keys()):
            self.emg_channel_order.extend(self.emg_sensor_map[sensor_idx])

        # IMU channels (sorted by sensor index, then axis order)
        self.imu_channel_order = []
        for sensor_idx in sorted(self.imu_sensor_map.keys()):
            self.imu_channel_order.extend(self.imu_sensor_map[sensor_idx])
    
    def _get_sensor_index(self, uuid):
        """
        Helper to extract sensor index from guid_to_name.
        Used to group EMG channels correctly.
        """
        name = self.guid_to_name.get(uuid, "")
        try:
            # Assumes format: "Sensor {index} | ..."
            return int(name.split()[1])
        except (IndexError, ValueError):
            return -1

    def _build_emg_groups(self):
        """
        Dynamically build EMG groups:
        - Single-channel Avanti sensors
        - 4-channel Galileo sensors (averaged across channels)
        """
        self.emg_groups = []
        used = set()

        # Simple heuristic: groups of 4 consecutive channels → Galileo, otherwise Avanti
        idx = 0
        while idx < len(self.emg_channel_order):
            remaining = len(self.emg_channel_order) - idx
            if remaining >= 4:
                # Check if next 4 channels belong to same sensor
                sensor_idxs = [self._get_sensor_index(uuid) for uuid in self.emg_channel_order[idx:idx+4]]
                if len(set(sensor_idxs)) == 1:
                    # Galileo group
                    self.emg_groups.append(self.emg_channel_order[idx:idx+4])
                    idx += 4
                    continue
            # Otherwise, single-channel Avanti
            self.emg_groups.append([self.emg_channel_order[idx]])
            idx += 1

    def _build_imu_groups(self):
        """
        Build IMU groups of 6 axes each (ACC-X,Y,Z + GYRO-X,Y,Z)
        """
        self.imu_groups = []
        n_axes = 6
        for i in range(0, len(self.imu_channel_order), n_axes):
            group = self.imu_channel_order[i:i+n_axes]
            if len(group) == n_axes:
                self.imu_groups.append(group)


    def _initialize_buffers(self):
        """Initialize deque buffers for all channels."""
        self.emg_buffers = {
            uuid: deque(maxlen=self.emg_win_size * 2)
            for uuid in self.emg_channel_order
        }

        self.imu_buffers = {
            uuid: deque(maxlen=self.imu_win_size * 2)
            for uuid in self.imu_channel_order
        }

    def add_raw_data(self, packet: Dict[str, np.ndarray]):
        """
        Add raw data packet to buffers.
        
        Args:
            packet: Dict mapping channel UUID to data array
        """
        with self.lock:
            for uuid, raw in packet.items():
                data = np.atleast_1d(raw)

                if uuid in self.emg_buffers:
                    self.emg_buffers[uuid].extend(data)
                    self.emg_sample_count += len(data)
                elif uuid in self.imu_buffers:
                    self.imu_buffers[uuid].extend(data)
                    self.imu_sample_count += len(data)

    def is_window_ready(self) -> bool:
        """Check if sufficient data is available for feature extraction."""
        # Check EMG buffers
        for buf in self.emg_buffers.values():
            if len(buf) < self.emg_win_size:
                return False

        # Check IMU buffers
        for buf in self.imu_buffers.values():
            if len(buf) < self.imu_win_size:
                return False

        return True

    def extract_features(self) -> Optional[np.ndarray]:
        """
        Extract features from current window using dynamic EMG/IMU groups.
        """
        with self.lock:
            if not self.is_window_ready():
                return None

            features = []

            # -----------------------
            # EMG FEATURES
            # -----------------------
            emg_feats_total = []
            for group in self.emg_groups:
                group_windows = [np.array(self.emg_buffers[uuid])[-self.emg_win_size:] for uuid in group]
                group_windows = np.array(group_windows)
                if group_windows.ndim > 1:
                    window = np.mean(group_windows, axis=0)  # average for Galileo
                else:
                    window = group_windows.flatten()  # Avanti
                feats = extract_emg_features(window, fs=self.fs_emg)
                emg_feats_total.extend(feats)
            features.extend(emg_feats_total)

            # -----------------------
            # IMU FEATURES
            # -----------------------
            imu_feats_total = []
            for group in self.imu_groups:
                imu_window = np.column_stack([np.array(self.imu_buffers[uuid])[-self.imu_win_size:] for uuid in group])
                feats = extract_imu_features(imu_window)
                imu_feats_total.extend(feats)
            features.extend(imu_feats_total)

            # -----------------------
            # DEBUG
            # -----------------------
            print(f"DEBUG: EMG features: {len(emg_feats_total)}, IMU features: {len(imu_feats_total)}, Total: {len(features)}")

            # Convert to array
            features_array = np.array(features).reshape(1, -1)

            # Validate
            if self.validate_features and features_array.shape[1] != self.expected_features:
                raise RuntimeError(
                    f"Feature mismatch: got {features_array.shape[1]}, expected {self.expected_features}"
                )

            # Scale
            features_scaled = self.scaler.transform(features_array)

            # Slide buffers
            self._slide_buffers()

        return features_scaled

    def _slide_buffers(self):
        """Slide window buffers forward by step size."""
        # Slide EMG buffers
        for uuid in self.emg_channel_order:
            buf = self.emg_buffers[uuid]
            for _ in range(min(self.emg_step, len(buf))):
                buf.popleft()
        self.emg_sample_count = max(self.emg_sample_count - self.emg_step, 0)

        # Slide IMU buffers
        for uuid in self.imu_channel_order:
            buf = self.imu_buffers[uuid]
            for _ in range(min(self.imu_step, len(buf))):
                buf.popleft()
        self.imu_sample_count = max(self.imu_sample_count - self.imu_step, 0)

    def predict(self) -> Optional[Tuple[int, np.ndarray, str]]:
        """
        Predict gesture from current window.
        
        Returns:
            (class_label, probabilities, class_name) or None if not ready
        """
        features = self.extract_features()
        if features is None:
            return None

        pred_class = self.clf.predict(features)[0]
        pred_probs = self.clf.predict_proba(features)[0]
        class_name = self.class_names.get(pred_class, f"Class {pred_class}")

        self.total_predictions += 1

        return pred_class, pred_probs, class_name

    def get_buffer_status(self) -> Dict:
        """Get current buffer status and statistics."""
        with self.lock:
            return {
                'emg_sample_count': self.emg_sample_count,
                'imu_sample_count': self.imu_sample_count,
                'emg_buffer_fill': min(len(buf) for buf in self.emg_buffers.values()) if self.emg_buffers else 0,
                'imu_buffer_fill': min(len(buf) for buf in self.imu_buffers.values()) if self.imu_buffers else 0,
                'window_ready': self.is_window_ready(),
                'total_predictions': self.total_predictions
            }

    def reset_buffers(self):
        """Clear all buffers and reset counters."""
        with self.lock:
            for buf in self.emg_buffers.values():
                buf.clear()
            for buf in self.imu_buffers.values():
                buf.clear()
            self.emg_sample_count = 0
            self.imu_sample_count = 0
            print("✓ Buffers reset")

    def get_channel_info(self) -> Dict:
        """Get information about configured channels."""
        return {
            'n_emg_sensors': len(self.emg_sensor_map),
            'n_imu_sensors': len(self.imu_sensor_map),
            'n_emg_channels': len(self.emg_channel_order),
            'n_imu_channels': len(self.imu_channel_order),
            'emg_sensors': {idx: len(uuids) for idx, uuids in self.emg_sensor_map.items()},
            'imu_sensors': list(self.imu_sensor_map.keys()),
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    print("""
IMPROVED REALTIME PROCESSOR
===========================

Key improvements:
1. Auto-detects expected feature count from scaler
2. Separate extract_features() and predict() methods
3. Better error messages and validation
4. Performance tracking
5. Buffer reset capability
6. Channel info reporting

Usage:
------
processor = RealtimeProcessor(
    delsys_client=client,
    model_path="models/my_model/trained_model.pkl",
    scaler_path="models/my_model/scaler.pkl",
    fs_emg=963,
    fs_imu=148.148,
    window_sec=0.2,
    overlap_sec=0.1
)

# In your streaming loop:
while True:
    data = client.poll_data()
    if data:
        processor.add_raw_data(data)
    
    if processor.is_window_ready():
        result = processor.predict()
        if result:
            label, probs, name = result
            print(f"{name}: {probs[label]*100:.1f}%")
    """)