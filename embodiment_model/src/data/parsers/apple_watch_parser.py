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
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# --- Record type → friendly name mapping ------------------------------------
# Add more HKQuantityTypeIdentifier values here as needed.

RECORD_TYPE_MAP = {
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN": "hrv_ms",
    "HKQuantityTypeIdentifierHeartRate":                "heart_rate_bpm",
    "HKQuantityTypeIdentifierElectrodermalActivity":    "eda_us",
    "HKQuantityTypeIdentifierRestingHeartRate":         "resting_hr_bpm",
    "HKQuantityTypeIdentifierRespiratoryRate":          "resp_rate_bpm",
    "HKQuantityTypeIdentifierOxygenSaturation":         "spo2_pct",
    "HKQuantityTypeIdentifierBodyTemperature":          "temp_c",
}

# Types used for embodiment model by default
EMBODIMENT_TYPES = {
    "HKQuantityTypeIdentifierHeartRateVariabilitySDNN",
    "HKQuantityTypeIdentifierHeartRate",
    "HKQuantityTypeIdentifierElectrodermalActivity",
    "HKQuantityTypeIdentifierRestingHeartRate",
    "HKQuantityTypeIdentifierRespiratoryRate",
}

DATE_FMT = "%Y-%m-%d %H:%M:%S %z"


# --- XML streaming parser ---------------------------------------------------

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
            raise ValueError(f"No export.xml found inside {path.name}")
        return zf.open(xml_names[0])
    else:
        return open(path, "r", encoding="utf-8")


# --- Public API -------------------------------------------------------------

def load(
    xml_path: str | Path,
    record_types: set[str] | None = None,
    source_filter: str | None = None,
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
        raise FileNotFoundError(f"Apple Health export not found: {xml_path}")

    if record_types is None:
        record_types = EMBODIMENT_TYPES

    logger.info("Streaming %s for types: %s", xml_path.name, record_types)

    rows: list[dict] = []
    seen_types: set[str] = set()

    with _open_xml(xml_path) as fh:
        for rec in _iter_records(fh):
            rtype = rec.get("type", "")
            if rtype not in record_types:
                continue

            source = rec.get("sourceName", "")
            if source_filter and source_filter.lower() not in source.lower():
                continue

            seen_types.add(rtype)
            try:
                value = float(rec.get("value", "nan"))
            except ValueError:
                value = float("nan")

            rows.append({
                "type":       RECORD_TYPE_MAP.get(rtype, rtype),
                "source":     source,
                "start_time": rec.get("startDate", ""),
                "end_time":   rec.get("endDate",   ""),
                "value":      value,
                "unit":       rec.get("unit", ""),
            })

    if not rows:
        raise ValueError(
            f"No records found for types {record_types} in {xml_path.name}. "
            f"Check that the export contains the expected data types."
        )

    df = pd.DataFrame(rows)

    # Parse timestamps
    df["start_time"] = pd.to_datetime(df["start_time"], format=DATE_FMT, utc=True, errors="coerce")
    df["end_time"]   = pd.to_datetime(df["end_time"],   format=DATE_FMT, utc=True, errors="coerce")

    df = df.sort_values("start_time").reset_index(drop=True)

    logger.info(
        "Loaded %d records | types found: %s | date range: %s → %s",
        len(df),
        {RECORD_TYPE_MAP.get(t, t) for t in seen_types},
        df["start_time"].min(),
        df["start_time"].max(),
    )
    return df


def pivot_to_timeseries(
    df: pd.DataFrame,
    resample_rule: str = "5s",
) -> pd.DataFrame:
    """
    Pivot the long-format DataFrame into a wide timeseries,
    resampled to a regular interval.

    Parameters
    ----------
    df : output of load()
    resample_rule : pandas offset alias, e.g. "1s", "5s", "1min"
        Apple Watch HRV is recorded every few minutes; HR every few seconds.
        "5s" is a reasonable compromise — missing values will be NaN.

    Returns
    -------
    Wide DataFrame indexed by timestamp, one column per data type.
    Use .interpolate() downstream for gap-filling if needed.
    """
    pivoted = (
        df.pivot_table(
            index="start_time",
            columns="type",
            values="value",
            aggfunc="mean",
        )
        .resample(resample_rule)
        .mean()
    )
    pivoted.index.name = "timestamp"
    return pivoted


def extract_session_window(
    df: pd.DataFrame,
    session_start: pd.Timestamp,
    session_end: pd.Timestamp,
) -> pd.DataFrame:
    """
    Filter records to a specific test session time window.

    Parameters
    ----------
    df : output of load()
    session_start / session_end : timezone-aware timestamps

    Returns
    -------
    Filtered and sorted DataFrame for that window.
    """
    mask = (df["start_time"] >= session_start) & (df["start_time"] <= session_end)
    windowed = df[mask].copy().reset_index(drop=True)
    logger.info(
        "Session window %s → %s: %d records",
        session_start.isoformat(), session_end.isoformat(), len(windowed),
    )
    return windowed


def get_available_types(xml_path: str | Path) -> dict[str, int]:
    """
    Scan the export to find all HK record types and their counts.
    Useful for exploring a new export before deciding what to load.

    Returns dict mapping friendly name (or raw type) → record count.
    """
    xml_path = Path(xml_path)
    counts: dict[str, int] = {}
    with _open_xml(xml_path) as fh:
        for rec in _iter_records(fh):
            rtype = rec.get("type", "unknown")
            friendly = RECORD_TYPE_MAP.get(rtype, rtype)
            counts[friendly] = counts.get(friendly, 0) + 1
    return dict(sorted(counts.items(), key=lambda x: -x[1]))


# --- Validation helper ------------------------------------------------------

def validate(df: pd.DataFrame) -> dict:
    """
    Sanity checks on a loaded Apple Watch DataFrame.
    Returns: {"ok": bool, "issues": [str]}
    """
    issues: list[str] = []

    if df.empty:
        return {"ok": False, "issues": ["DataFrame is empty"]}

    # NaN timestamps
    if df["start_time"].isna().any():
        issues.append("Some start_time values could not be parsed")

    # NaN values
    nan_frac = df["value"].isna().mean()
    if nan_frac > 0.05:
        issues.append(f"{nan_frac:.1%} of values are NaN")

    # Physical range checks per type
    checks = {
        "hrv_ms":          (0, 300),
        "heart_rate_bpm":  (30, 220),
        "eda_us":          (0, 100),
        "resting_hr_bpm":  (30, 120),
        "resp_rate_bpm":   (4, 40),
    }
    for dtype, (lo, hi) in checks.items():
        sub = df[df["type"] == dtype]["value"].dropna()
        if sub.empty:
            continue
        out_of_range = ((sub < lo) | (sub > hi)).sum()
        if out_of_range:
            issues.append(
                f"{dtype}: {out_of_range} value(s) outside expected range [{lo}, {hi}]"
            )

    return {"ok": len(issues) == 0, "issues": issues}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else "export.xml"
    df = load(path)
    print(df.head(10).to_string())
    print("\nShape:", df.shape)
    print("\nRecord counts by type:")
    print(df.groupby("type").size().to_string())
    result = validate(df)
    print("\nValidation:", result)