"""
bioradio_parser.py
------------------
Parses BioRadio .bcrx files into a pandas DataFrame.

.bcrx structure (ZIP archive):
  header.xml   — channel config, sample rates, recording metadata
  <N>_<M>.rec  — compressed binary sample data

Observed .rec format:
  raw DEFLATE → little-endian IEEE-754 float32 samples

The decoded float values are already in the physical units specified
by the header (e.g. µS for GSR/EDA).
"""

from __future__ import annotations

import logging
import zipfile
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --- Data classes -----------------------------------------------------------

@dataclass
class ChannelConfig:
    name: str
    units: str
    sample_rate_hz: int
    bit_resolution: int
    gain: float
    offset: float
    phys_min: float
    phys_max: float
    raw_min: int
    raw_max: int
    enabled: bool


@dataclass
class BioRadioSession:
    recording_start: datetime
    recording_length_s: float
    total_packets: int
    missing_packets: int
    channels: list[ChannelConfig] = field(default_factory=list)

    @property
    def enabled_channels(self) -> list[ChannelConfig]:
        return [c for c in self.channels if c.enabled]


# --- XML header parsing -----------------------------------------------------

def _parse_header(xml_bytes: bytes) -> BioRadioSession:
    """Parse header.xml from a .bcrx archive."""
    root = ET.fromstring(xml_bytes)

    start_str = root.findtext("RecordingStartTime", "").strip()
    recording_start = (
        datetime.fromisoformat(start_str)
        if start_str else datetime.now(timezone.utc)
    )

    ticks = int(root.findtext("RecordingLengthTicks", "0"))
    length_s = ticks / 10_000_000

    session = BioRadioSession(
        recording_start=recording_start,
        recording_length_s=length_s,
        total_packets=int(root.findtext("TotalPackets", "0")),
        missing_packets=int(root.findtext("MissingPacketCount", "0")),
    )

    hw_lookup = {}
    for ch_cfg in root.findall(".//BioPotentialChannelConfiguration"):
        name = ch_cfg.findtext("n", "").strip()
        hw_lookup[name] = {
            "bit_res": int(ch_cfg.findtext("BitResolution", "24")),
            "gain": float(ch_cfg.findtext("Gain", "1")),
        }

    for sig_def in root.findall(".//CustomSignalDefinition"):
        sig = sig_def.find("Signal")
        if sig is None:
            continue

        name = (sig.findtext("Name") or sig.findtext("n") or "").strip()
        units = sig.findtext("Units", "").strip()
        sps = int(sig.findtext("SamplesPerSecond", "0"))
        enabled = sig.findtext("Enabled", "false").strip().lower() == "true"

        hw = hw_lookup.get(name, {})

        session.channels.append(ChannelConfig(
            name=name,
            units=units,
            sample_rate_hz=sps,
            bit_resolution=hw.get("bit_res", 24),
            gain=float(sig_def.findtext("Gain", "1")),
            offset=float(sig_def.findtext("Offset", "0")),
            phys_min=float(sig.findtext("MinValue", "0")),
            phys_max=float(sig.findtext("MaxValue", "0")),
            raw_min=int(sig.findtext("RawMinValue", "0")),
            raw_max=int(sig.findtext("RawMaxValue", "0")),
            enabled=enabled,
        ))

    logger.debug(
        "Header parsed: %d channel(s), recording start %s, %.2fs",
        len(session.channels), recording_start.isoformat(), length_s,
    )
    return session


# --- Binary .rec decoding ---------------------------------------------------

def _decode_rec(rec_bytes: bytes) -> np.ndarray:
    """
    Decompress and decode a .rec segment.

    Observed format:
      raw DEFLATE → little-endian float32 samples
    """
    try:
        decompressed = zlib.decompress(rec_bytes, -zlib.MAX_WBITS)
    except zlib.error as e:
        raise ValueError(f"Failed to decompress .rec data: {e}") from e

    if len(decompressed) % 4 != 0:
        raise ValueError(
            f"Decompressed .rec size ({len(decompressed)}) "
            "is not divisible by 4"
        )

    return np.frombuffer(decompressed, dtype="<f4").astype(np.float64)


# --- Public API -------------------------------------------------------------

def load(bcrx_path: str | Path) -> pd.DataFrame:
    """Load a .bcrx file into a pandas DataFrame."""
    bcrx_path = Path(bcrx_path)

    if not bcrx_path.exists():
        raise FileNotFoundError(f"BioRadio file not found: {bcrx_path}")

    with zipfile.ZipFile(bcrx_path, "r") as zf:
        names = zf.namelist()

        if "header.xml" not in names:
            raise ValueError(f"No header.xml inside {bcrx_path.name}")

        session = _parse_header(zf.read("header.xml"))

        if not session.enabled_channels:
            raise ValueError("No enabled channels found in header.xml")

        rec_files = sorted(n for n in names if n.lower().endswith(".rec"))
        if not rec_files:
            raise ValueError(f"No .rec data files inside {bcrx_path.name}")

        # Each .rec file is its own compressed stream.
        samples = [
            _decode_rec(zf.read(rec_file))
            for rec_file in rec_files
        ]

    channel = session.enabled_channels[0]
    physical = np.concatenate(samples)

    if len(physical) == 0:
        raise ValueError("No sample data decoded from .rec file(s)")

    expected = session.recording_length_s * channel.sample_rate_hz
    if abs(len(physical) - expected) > 1:
        logger.warning(
            "Decoded %d samples, expected %.1f from header",
            len(physical), expected,
        )

    dt_s = 1.0 / channel.sample_rate_hz
    elapsed = np.arange(len(physical)) * dt_s
    timestamps = (
        pd.Timestamp(session.recording_start)
        + pd.to_timedelta(elapsed, unit="s")
    )

    df = pd.DataFrame({
        "timestamp": timestamps,
        "elapsed_s": elapsed,
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
    """Run basic sanity checks on a loaded BioRadio DataFrame."""
    issues = []

    if df.empty:
        return {"ok": False, "issues": ["DataFrame is empty"]}

    data_cols = [c for c in df.columns if c not in ("timestamp", "elapsed_s")]

    for col in data_cols:
        if df[col].isna().any():
            issues.append(f"{col}: contains NaN values")
        if np.isinf(df[col]).any():
            issues.append(f"{col}: contains infinite values")

    if not df["timestamp"].is_monotonic_increasing:
        issues.append("timestamps are not monotonically increasing")

    if len(df) > 1:
        diffs = df["elapsed_s"].diff().dropna()
        expected_dt = diffs.median()

        if expected_dt <= 0:
            issues.append("invalid sample interval")

    return {"ok": len(issues) == 0, "issues": issues}


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    path = sys.argv[1] if len(sys.argv) > 1 else "bioradio1.bcrx"
    df = load(path)

    print(df.head(10).to_string())
    print("\nShape:", df.shape)
    print("Validation:", validate(df))