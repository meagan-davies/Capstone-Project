"""
bioradio_parser.py
------------------
Parses BioRadio .bcrx files into a pandas DataFrame.

.bcrx structure (ZIP archive):
  header.xml   — channel config, sample rates, recording metadata
  <N>_<M>.rec  — binary sample data (N=channel group, M=segment)

Binary packet format (22 bytes per packet):
  Bytes 0-1  : packet sequence number (uint16, little-endian)
  Bytes 2-3  : status word (uint16, little-endian)
  Bytes 4-21 : sample data — up to 6 x 24-bit signed ints (little-endian)

Physical scaling (per header):
  value_µS = ((raw - raw_min) / (raw_max - raw_min)) * (phys_max - phys_min) * gain + offset + phys_min
"""

from __future__ import annotations

import struct
import zipfile
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# --- Packet constants -------------------------------------------------------
PACKET_BYTES   = 22
HEADER_BYTES   = 4          # seq(2) + status(2)
SAMPLE_BYTES   = 3          # 24-bit signed
SAMPLES_PER_PACKET = (PACKET_BYTES - HEADER_BYTES) // SAMPLE_BYTES  # = 6


# --- Data classes -----------------------------------------------------------

@dataclass
class ChannelConfig:
    name:           str
    units:          str
    sample_rate_hz: int
    bit_resolution: int
    gain:           float
    offset:         float
    phys_min:       float
    phys_max:       float
    raw_min:        int
    raw_max:        int
    enabled:        bool


@dataclass
class BioRadioSession:
    recording_start:  datetime
    recording_length_s: float
    total_packets:    int
    missing_packets:  int
    channels:         list[ChannelConfig] = field(default_factory=list)

    @property
    def enabled_channels(self) -> list[ChannelConfig]:
        return [c for c in self.channels if c.enabled]


# --- XML header parsing -----------------------------------------------------

def _parse_header(xml_bytes: bytes) -> BioRadioSession:
    """Parse header.xml from a .bcrx archive into a BioRadioSession."""
    root = ET.fromstring(xml_bytes)
    ns = {"xsi": "http://www.w3.org/2001/XMLSchema-instance"}

    # Recording-level metadata
    start_str = root.findtext("RecordingStartTime", "").strip()
    # ISO-8601 with offset e.g. "2026-01-27T13:33:16.240-07:00"
    recording_start = datetime.fromisoformat(start_str) if start_str else datetime.now(timezone.utc)

    ticks = int(root.findtext("RecordingLengthTicks", "0"))
    length_s = ticks / 10_000_000  # .NET ticks = 100 ns each

    total_packets   = int(root.findtext("TotalPackets", "0"))
    missing_packets = int(root.findtext("MissingPacketCount", "0"))

    session = BioRadioSession(
        recording_start=recording_start,
        recording_length_s=length_s,
        total_packets=total_packets,
        missing_packets=missing_packets,
    )

    # Channel definitions come from two places:
    #   1. CustomSignalDefinitions — describes each enabled signal with scaling
    #   2. DeviceConfig/BioPotentialChannels — per-channel hardware config

    # Build a lookup from channel index → hardware config
    hw_lookup: dict[int, dict] = {}
    for ch_cfg in root.findall(".//BioPotentialChannelConfiguration"):
        idx  = int(ch_cfg.findtext("ChannelIndex", "0"))
        enabled = ch_cfg.findtext("Enabled", "false").strip().lower() == "true"
        hw_lookup[idx] = {
            "name":    ch_cfg.findtext("n", "").strip(),
            "enabled": enabled,
            "bit_res": int(ch_cfg.findtext("BitResolution", "16")),
            "gain":    float(ch_cfg.findtext("Gain", "1")),
        }

    # Parse each CustomSignalDefinition
    for sig_def in root.findall(".//CustomSignalDefinition"):
        sig = sig_def.find("Signal")
        if sig is None:
            continue

        # Header uses <Name> (capital N) inside <Signal>, not <n>
        name    = (sig.findtext("Name") or sig.findtext("n") or "").strip()
        units   = sig.findtext("Units", "").strip()
        sps     = int(sig.findtext("SamplesPerSecond", "500"))
        enabled = sig.findtext("Enabled", "false").strip().lower() == "true"

        phys_max = float(sig.findtext("MaxValue", "1"))
        phys_min = float(sig.findtext("MinValue", "0"))
        raw_min  = int(sig.findtext("RawMinValue", "-8388608"))
        raw_max  = int(sig.findtext("RawMaxValue",  "8388607"))

        # Scaling from the signal definition level
        gain   = float(sig_def.findtext("Gain",   "1"))
        offset = float(sig_def.findtext("Offset", "0"))

        # Find matching hardware config by name to get bit resolution
        hw = next((v for v in hw_lookup.values() if v["name"] == name), {})
        bit_res = hw.get("bit_res", 24)

        session.channels.append(ChannelConfig(
            name=name,
            units=units,
            sample_rate_hz=sps,
            bit_resolution=bit_res,
            gain=gain,
            offset=offset,
            phys_min=phys_min,
            phys_max=phys_max,
            raw_min=raw_min,
            raw_max=raw_max,
            enabled=enabled,
        ))

    logger.debug(
        "Header parsed: %d channel(s), recording start %s, %.2fs",
        len(session.channels), recording_start.isoformat(), length_s,
    )
    return session


# --- Binary .rec decoding ---------------------------------------------------

def _decode_rec(rec_bytes: bytes, channel: ChannelConfig) -> np.ndarray:
    """
    Decode raw .rec bytes into physical-unit float samples.

    Packet layout (22 bytes):
      [0:2]  seq    uint16 LE
      [2:4]  status uint16 LE
      [4:22] data   6 × 24-bit signed LE integers

    Physical conversion:
      raw_range = raw_max - raw_min  (or 2^(bit_res-1) * 2 when header values are sentinel)
      phys = ((raw - raw_min) / raw_range) * (phys_max - phys_min) * gain + offset + phys_min
    """
    n_packets = len(rec_bytes) // PACKET_BYTES
    if n_packets == 0:
        return np.array([], dtype=np.float64)

    raw_samples: list[int] = []
    for i in range(n_packets):
        start = i * PACKET_BYTES
        data_slice = rec_bytes[start + HEADER_BYTES: start + PACKET_BYTES]
        for j in range(SAMPLES_PER_PACKET):
            chunk = data_slice[j * SAMPLE_BYTES: (j + 1) * SAMPLE_BYTES]
            raw = int.from_bytes(chunk, byteorder="little", signed=True)
            raw_samples.append(raw)

    arr = np.array(raw_samples, dtype=np.float64)

    # Determine raw range (header may use sentinel value −8388608 for both min/max)
    raw_min = channel.raw_min
    raw_max = channel.raw_max
    if raw_min == raw_max:  # sentinel: use full 24-bit signed range
        raw_min = -(2 ** (channel.bit_resolution - 1))
        raw_max =  (2 ** (channel.bit_resolution - 1)) - 1

    raw_range = raw_max - raw_min
    if raw_range == 0:
        raw_range = 1  # guard

    phys_range = channel.phys_max - channel.phys_min
    physical = (
        ((arr - raw_min) / raw_range) * phys_range * channel.gain
        + channel.offset
        + channel.phys_min
    )

    logger.debug(
        "Decoded %d raw samples from %d packets → %d physical values",
        len(arr), n_packets, len(physical),
    )
    return physical


# --- Public API -------------------------------------------------------------

def load(bcrx_path: str | Path) -> pd.DataFrame:
    """
    Load a .bcrx file and return a tidy DataFrame.

    Returns
    -------
    DataFrame with columns:
        timestamp   — absolute datetime (UTC) for each sample
        elapsed_s   — seconds since recording start
        <channel>   — physical-unit value (e.g. SKN in µS for EDA/GSR)
        ... one column per enabled channel

    Raises
    ------
    FileNotFoundError  if the path does not exist
    ValueError         if no enabled channels or no .rec files found
    """
    bcrx_path = Path(bcrx_path)
    if not bcrx_path.exists():
        raise FileNotFoundError(f"BioRadio file not found: {bcrx_path}")

    with zipfile.ZipFile(bcrx_path, "r") as zf:
        names = zf.namelist()

        # --- Parse header ---
        if "header.xml" not in names:
            raise ValueError(f"No header.xml inside {bcrx_path.name}")
        session = _parse_header(zf.read("header.xml"))

        if not session.enabled_channels:
            raise ValueError("No enabled channels found in header.xml")

        # --- Find .rec files (sorted for multi-segment recordings) ---
        rec_files = sorted(n for n in names if n.endswith(".rec"))
        if not rec_files:
            raise ValueError(f"No .rec data files inside {bcrx_path.name}")

        # Concatenate all segments in order
        rec_bytes = b"".join(zf.read(r) for r in rec_files)

    logger.info(
        "%s: %d enabled channel(s), %d .rec segment(s), %.2f s recording",
        bcrx_path.name, len(session.enabled_channels), len(rec_files),
        session.recording_length_s,
    )

    # For now decode against the first enabled channel (SKN/EDA typically)
    # Multi-channel interleaving logic can be added here when needed.
    channel = session.enabled_channels[0]
    physical = _decode_rec(rec_bytes, channel)

    if len(physical) == 0:
        raise ValueError("No sample data decoded from .rec file(s)")

    # Build time index from recording start + sample period
    dt_s   = 1.0 / channel.sample_rate_hz
    elapsed = np.arange(len(physical)) * dt_s
    timestamps = pd.to_datetime(
        [session.recording_start.timestamp() + e for e in elapsed],
        unit="s", utc=True,
    )

    df = pd.DataFrame({
        "timestamp":  timestamps,
        "elapsed_s":  elapsed,
        channel.name: physical,
    })

    logger.info(
        "Loaded %d samples of %s (%s) @ %d Hz | %.3f – %.3f %s",
        len(df), channel.name, channel.units,
        channel.sample_rate_hz,
        physical.min(), physical.max(), channel.units,
    )
    return df


# --- Validation helper ------------------------------------------------------

def validate(df: pd.DataFrame) -> dict:
    """
    Quick sanity checks on a loaded BioRadio DataFrame.

    Returns a dict with keys: ok (bool), issues (list[str])
    """
    issues: list[str] = []

    if df.empty:
        issues.append("DataFrame is empty")
        return {"ok": False, "issues": issues}

    data_cols = [c for c in df.columns if c not in ("timestamp", "elapsed_s")]

    for col in data_cols:
        if df[col].isna().any():
            issues.append(f"{col}: contains NaN values")
        if (df[col] < 0).any():
            issues.append(f"{col}: negative physical values (check scaling)")

    # Check for timestamp monotonicity
    if not df["timestamp"].is_monotonic_increasing:
        issues.append("timestamps are not monotonically increasing")

    # Check for large gaps (> 2× expected sample period)
    if len(df) > 1:
        diffs = df["elapsed_s"].diff().dropna()
        expected_dt = diffs.median()
        big_gaps = (diffs > expected_dt * 2).sum()
        if big_gaps:
            issues.append(f"{big_gaps} timing gap(s) detected (> 2× sample period)")

    return {"ok": len(issues) == 0, "issues": issues}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else "bioradio1.bcrx"
    df = load(path)
    print(df.head(10).to_string())
    print("\nShape:", df.shape)
    result = validate(df)
    print("Validation:", result)