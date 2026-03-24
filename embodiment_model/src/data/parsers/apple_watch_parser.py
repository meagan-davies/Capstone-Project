"""
apple_watch_parser.py
---------------------
Parses Apple Health export.xml files into pandas DataFrames.

Apple Health export structure:
  export.xml  (or export.zip containing export.xml)
    <HealthData locale="...">
      <ExportDate value="..."/>
      <Record type="HKQuantityTypeIdentifier..." sourceName="..." startDate="..." endDate="..." value="..." unit="..."/>
      ...
    </HealthData>

Records relevant to embodiment model:
  HKQuantityTypeIdentifierHeartRateVariabilitySDNN  — HRV (ms)
  HKQuantityTypeIdentifierHeartRate                 — Heart rate (bpm)
  HKQuantityTypeIdentifierElectrodermalActivity     — EDA / skin conductance (µS)
  HKQuantityTypeIdentifierRestingHeartRate          — Resting HR (bpm)
  HKQuantityTypeIdentifierRespiratoryRate           — Respiratory rate (breaths/min)

Note: EDA is only available on Apple Watch Series 4+ with certain health conditions enabled,
or via third-party apps that write to HealthKit.

Date format in export.xml: "2024-03-15 14:23:00 +0000"
"""

from __future__ import annotations

import zipfile
import logging
from pathlib import Path
from xml.etree.ElementTree import iterparse
from typing import Iterator, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Type mapping
# ---------------------------------------------------------------------------

RECORD_TYPE_MAP = {
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv_ms",
    "HKQuantityTypeIdentifierHeartRate":                "heart_rate_bpm",
    "HKQuantityTypeIdentifierRestingHeartRate":         "resting_hr_bpm",
    "HKQuantityTypeIdentifierRespiratoryRate":          "resp_rate_bpm",
    "HKQuantityTypeIdentifierOxygenSaturation":         "spo2_pct",
    "HKQuantityTypeIdentifierBodyTemperature":          "temp_c",
    # NOTE: EDA removed (not supported natively by Apple Watch)
}

DATE_FMT = "%Y-%m-%d %H:%M:%S %z"


# ---------------------------------------------------------------------------
# Streaming XML
# ---------------------------------------------------------------------------

def _iter_records(xml_source) -> Iterator[dict]:
    """
    Stream <Record> elements from a file-like object using iterparse.
    Yields one dict per record (attributes only — no child elements needed
    for the quantity types we care about).
    Uses iterparse to avoid loading the full XML tree into memory.
    """
    for event, elem in iterparse(xml_source, events=("end",)):
        if elem.tag == "Record":
            yield elem.attrib
            elem.clear()  # free memory immediately


def _open_xml(path: Path):
    """
    Return a file-like object for the XML data.
    Handles both raw export.xml and zipped export.zip.
    """
    if path.suffix.lower() == ".zip":
        zf = zipfile.ZipFile(path, "r")
        # Apple exports as 'apple_health_export/export.xml' inside the zip
        xml_names = [n for n in zf.namelist() if n.endswith("export.xml")]
        if not xml_names:
            raise ValueError(f"No export.xml found in {path}")
        return zf.open(xml_names[0])
    else:
        return open(path, "r", encoding="utf-8")


# ---------------------------------------------------------------------------
# Core loader
# ---------------------------------------------------------------------------

def load(
    xml_path: str | Path,
    record_types: Optional[set[str]] = None,
    source_filter: Optional[str] = None,
    session_start: Optional[pd.Timestamp] = None,
    session_end: Optional[pd.Timestamp] = None,
    max_records: Optional[int] = None,   # debug control
) -> pd.DataFrame:
    """
    Load an Apple Health export XML and return a tidy DataFrame.

    Parameters
    ----------
    xml_path : path to export.xml or export.zip
    record_types : set of HKQuantityTypeIdentifier strings to include.
        Defaults to EMBODIMENT_TYPES (HRV, HR, EDA, resting HR, resp rate).
    source_filter : if set, only include records where sourceName contains
        this string (e.g. "Apple Watch" to exclude iPhone records).

    Returns
    -------
    DataFrame with columns:
        type         — friendly name (from RECORD_TYPE_MAP, or raw HK type)
        source       — sourceName from the record
        start_time   — pd.Timestamp (UTC)
        end_time     — pd.Timestamp (UTC)
        value        — float
        unit         — unit string from the record

    Raises
    ------
    FileNotFoundError  if the path does not exist
    ValueError         if no records are found for the requested types
    """
    xml_path = Path(xml_path)
    if not xml_path.exists():
        raise FileNotFoundError(xml_path)

    logger.info("Loading Apple Health: %s", xml_path.name)

    rows = []
    type_counts = {}
    total_seen = 0

    with _open_xml(xml_path) as fh:
        for rec in _iter_records(fh):
            total_seen += 1

            rtype = rec.get("type", "")
            type_counts[rtype] = type_counts.get(rtype, 0) + 1

            # Type filtering (optional)
            if record_types is not None and rtype not in record_types:
                continue

            source = rec.get("sourceName", "")

            # FIX: substring filter
            if source_filter and source_filter.lower() not in source.lower():
                continue

            try:
                value = float(rec.get("value", "nan"))
            except ValueError:
                value = np.nan

            rows.append({
                "type_raw":   rtype,
                "type":       RECORD_TYPE_MAP.get(rtype, rtype),
                "source":     source,
                "start_time": rec.get("startDate"),
                "end_time":   rec.get("endDate"),
                "value":      value,
                "unit":       rec.get("unit", ""),
            })

            if max_records and len(rows) >= max_records:
                break

    if not rows:
        logger.warning("No matching records after filtering")
        return pd.DataFrame()

    df = pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Timestamp handling (critical fix)
    # -----------------------------------------------------------------------
    df["start_time"] = pd.to_datetime(df["start_time"], format=DATE_FMT, utc=True, errors="coerce")
    df["end_time"]   = pd.to_datetime(df["end_time"],   format=DATE_FMT, utc=True, errors="coerce")

    df = df.dropna(subset=["start_time"]).sort_values("start_time")

    # -----------------------------------------------------------------------
    # Session alignment
    # -----------------------------------------------------------------------
    if session_start is not None:
        session_start = pd.Timestamp(session_start, tz="UTC")

        df["elapsed_s"] = (df["start_time"] - session_start).dt.total_seconds()

        if session_end is not None:
            session_end = pd.Timestamp(session_end, tz="UTC")
            df = df[(df["start_time"] >= session_start) & (df["start_time"] <= session_end)]

    else:
        # fallback: relative to first sample
        t0 = df["start_time"].iloc[0]
        df["elapsed_s"] = (df["start_time"] - t0).dt.total_seconds()

    df = df.reset_index(drop=True)

    # -----------------------------------------------------------------------
    # Diagnostics
    # -----------------------------------------------------------------------
    detected = {RECORD_TYPE_MAP.get(k, k): v for k, v in type_counts.items()}

    logger.info("Total records scanned: %d", total_seen)
    logger.info("Detected types: %s", detected)
    logger.info("Loaded rows: %d", len(df))

    if record_types:
        missing = set(record_types) - set(type_counts.keys())
        if missing:
            logger.warning("Missing requested types: %s", missing)

    return df


# ---------------------------------------------------------------------------
# Timeseries conversion
# ---------------------------------------------------------------------------

def pivot_to_timeseries(
    df: pd.DataFrame,
    resample_rule: str = "5s",
    interpolate: bool = False,
) -> pd.DataFrame:

    ts = (
        df.pivot_table(
            index="start_time",
            columns="type",
            values="value",
            aggfunc="mean",
        )
        .sort_index()
        .resample(resample_rule)
        .mean()
    )

    if interpolate:
        ts = ts.interpolate(limit_direction="both")

    ts.index.name = "timestamp"
    return ts


# ---------------------------------------------------------------------------
# Exploration utility (fast scan)
# ---------------------------------------------------------------------------

def scan_types(xml_path: str | Path, limit: int = 1_000_000) -> dict:
    """
    Fast scan to understand available signals before loading.
    """
    xml_path = Path(xml_path)
    counts = {}

    with _open_xml(xml_path) as fh:
        for i, rec in enumerate(_iter_records(fh)):
            rtype = rec.get("type", "unknown")
            counts[rtype] = counts.get(rtype, 0) + 1
            if i >= limit:
                break

    return dict(sorted(counts.items(), key=lambda x: -x[1]))

def validate(df: pd.DataFrame) -> dict:
    """
    Validate Apple Watch dataframe.

    Returns:
        {"ok": bool, "issues": [str]}
    """
    issues = []

    if df is None:
        return {"ok": True, "issues": ["watch data not loaded"]}

    if df.empty:
        return {"ok": True, "issues": ["watch dataframe empty (no matching records)"]}

    # Timestamp integrity
    if df["start_time"].isna().any():
        issues.append("Invalid start_time values")

    # Value quality
    nan_frac = df["value"].isna().mean()
    if nan_frac > 0.1:
        issues.append(f"{nan_frac:.1%} NaN values")

    # Basic physiological sanity checks
    bounds = {
        "heart_rate_bpm": (30, 220),
        "hrv_ms": (0, 300),
        "resp_rate_bpm": (4, 40),
        "spo2_pct": (70, 100),
    }

    for t, (lo, hi) in bounds.items():
        sub = df[df["type"] == t]["value"].dropna()
        if not sub.empty:
            out = ((sub < lo) | (sub > hi)).sum()
            if out > 0:
                issues.append(f"{t}: {out} out-of-range values")

    return {"ok": len(issues) == 0, "issues": issues}