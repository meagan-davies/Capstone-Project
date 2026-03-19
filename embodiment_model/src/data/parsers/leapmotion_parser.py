"""
leapmotion_parser.py
---------------------
Parses Leap Motion Controller CSV exports into a clean pandas DataFrame.

CSV format (tab-separated, no timestamp column):
  - One row per frame captured at ~60 Hz (configurable)
  - Columns follow the pattern:
      hands, fingers
      hand0:tracking, hand0:confidence, hand0:type
      hand0/palm:tx/ty/tz, hand0/palm:rx/ry/rz
      hand0/wrist:tx/ty/tz
      hand0/elbow:tracking, hand0/elbow:tx/ty/tz, hand0/elbow:rx/ry/rz
      hand0:vx/vy/vz
      hand0:pinch, hand0:grab
      hand0/fingerN:tracking, hand0/fingerN:tx/ty/tz, hand0/fingerN_nml:tx/ty/tz
      ... (fingerN for N = 0..4, though file may stop at finger2 or finger4)

Coordinate system: Leap Motion right-handed (x=right, y=up, z=toward user), mm
Angles: degrees
Pinch/Grab: 0.0 (open) → 1.0 (closed)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default Leap Motion frame rate when no timestamp column is present
DEFAULT_FRAME_RATE_HZ: float = 60.0


# --- Column groups for easy downstream access -------------------------------

PALM_COLS       = ["palm_tx", "palm_ty", "palm_tz",
                   "palm_rx", "palm_ry", "palm_rz"]
WRIST_COLS      = ["wrist_tx", "wrist_ty", "wrist_tz"]
ELBOW_COLS      = ["elbow_tx", "elbow_ty", "elbow_tz",
                   "elbow_rx", "elbow_ry", "elbow_rz"]
VELOCITY_COLS   = ["vel_x", "vel_y", "vel_z"]
GESTURE_COLS    = ["pinch", "grab"]


def _finger_cols(n: int) -> list[str]:
    """Return standardised column names for finger N (0-indexed)."""
    return [
        f"finger{n}_tx", f"finger{n}_ty", f"finger{n}_tz",
        f"finger{n}_nml_tx", f"finger{n}_nml_ty", f"finger{n}_nml_tz",
    ]


# --- Raw column → clean name map -------------------------------------------

def _build_rename_map(raw_cols: list[str]) -> dict[str, str]:
    """
    Map raw Leap Motion column names to clean snake_case equivalents.

    Examples:
        "hand0/palm:tx"         → "palm_tx"
        "hand0:pinch"           → "pinch"
        "hand0/finger2_nml:ty"  → "finger2_nml_ty"
        "hand0/elbow:rx"        → "elbow_rx"
    """
    rename: dict[str, str] = {}
    for col in raw_cols:
        clean = (
            col
            .replace("hand0/palm:", "palm_")
            .replace("hand0/wrist:", "wrist_")
            .replace("hand0/elbow:", "elbow_")
            .replace("hand0:", "")          # covers tracking, confidence, type, vx, pinch...
            .replace("hand0/", "")          # catches finger sub-paths
            .replace("/", "_")
            .replace(":", "_")
            # normalise velocity axes:  vx → vel_x
            .replace("vx", "vel_x")
            .replace("vy", "vel_y")
            .replace("vz", "vel_z")
            # normalise finger normals: finger0_nml:tx → finger0_nml_tx
            # already handled by the colon replacement above
        )
        # drop trailing underscore if any
        clean = clean.strip("_")
        rename[col] = clean
    return rename


# --- Public API -------------------------------------------------------------

def load(
    csv_path: str | Path,
    frame_rate_hz: float = DEFAULT_FRAME_RATE_HZ,
    session_start: Optional[pd.Timestamp] = None,
    hand_index: int = 0,
) -> pd.DataFrame:
    """
    Load a Leap Motion CSV and return a clean, tidy DataFrame.

    Parameters
    ----------
    csv_path : path to the .csv file
    frame_rate_hz : frames per second to use for timestamp synthesis
        (Leap Motion does not embed timestamps in the CSV export)
    session_start : optional absolute start time; if None, elapsed_s only
    hand_index : which hand to parse (0 = first detected hand)

    Returns
    -------
    DataFrame with columns:
        frame           — 0-indexed frame number
        elapsed_s       — seconds since frame 0
        timestamp       — absolute datetime (if session_start provided)
        tracking        — hand tracking confidence flag (0/1)
        confidence      — float [0, 1]
        hand_type       — int (0=left?, 1=?, 2=right? — Leap encoding)
        palm_tx/ty/tz   — palm position (mm)
        palm_rx/ry/rz   — palm orientation (degrees)
        wrist_tx/ty/tz  — wrist position (mm)
        elbow_tx/ty/tz  — elbow position (mm)
        elbow_rx/ry/rz  — elbow orientation (degrees)
        vel_x/y/z       — hand velocity (mm/s)
        pinch           — pinch strength [0, 1]
        grab            — grab strength [0, 1]
        finger0–4_tx/ty/tz       — fingertip positions (mm)
        finger0–4_nml_tx/ty/tz   — fingertip normals

    Raises
    ------
    FileNotFoundError  if the path does not exist
    ValueError         if required columns are missing
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Leap Motion CSV not found: {csv_path}")

    # --- Read raw -----------------------------------------------------------
    raw = pd.read_csv(csv_path, sep="\t")
    logger.info("%s: %d frames, %d raw columns", csv_path.name, len(raw), len(raw.columns))

    # --- Rename columns -----------------------------------------------------
    rename_map = _build_rename_map(list(raw.columns))
    df = raw.rename(columns=rename_map)

    # --- Validate required columns ------------------------------------------
    required = ["pinch", "grab", "palm_tx", "palm_ty", "palm_tz"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required Leap columns after rename: {missing}")

    # --- Synthesise time axis -----------------------------------------------
    n_frames = len(df)
    dt_s     = 1.0 / frame_rate_hz
    elapsed  = np.arange(n_frames, dtype=np.float64) * dt_s

    df.insert(0, "frame",     np.arange(n_frames))
    df.insert(1, "elapsed_s", elapsed)

    if session_start is not None:
        ts = pd.to_datetime(session_start) + pd.to_timedelta(elapsed, unit="s")
        df.insert(2, "timestamp", ts)

    # --- Rename remaining tracking/meta cols for clarity --------------------
    renames_post = {}
    if "tracking"   in df.columns: renames_post["tracking"]   = "tracking"
    if "confidence" in df.columns: renames_post["confidence"] = "confidence"
    if "type"       in df.columns: renames_post["type"]       = "hand_type"
    if renames_post:
        df = df.rename(columns=renames_post)

    # --- Cast types ---------------------------------------------------------
    for col in ["frame", "hands", "fingers"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    for col in ["pinch", "grab", "confidence"]:
        if col in df.columns:
            df[col] = df[col].astype(np.float32)

    logger.info(
        "Loaded %d frames | pinch [%.3f, %.3f] | grab [%.3f, %.3f]",
        len(df),
        df["pinch"].min(), df["pinch"].max(),
        df["grab"].min(),  df["grab"].max(),
    )
    return df


# --- Derived features -------------------------------------------------------

def compute_velocity_magnitude(df: pd.DataFrame) -> pd.Series:
    """Return |velocity| in mm/s for each frame."""
    return np.sqrt(df["vel_x"]**2 + df["vel_y"]**2 + df["vel_z"]**2)


def compute_palm_trajectory(df: pd.DataFrame) -> pd.DataFrame:
    """
    Return a minimal trajectory DataFrame: elapsed_s + palm XYZ + speed.
    Useful for control_accuracy metrics.
    """
    out = df[["elapsed_s", "palm_tx", "palm_ty", "palm_tz"]].copy()
    out["speed_mm_s"] = compute_velocity_magnitude(df)
    return out


def detect_pinch_events(
    df: pd.DataFrame,
    threshold: float = 0.7,
    min_duration_frames: int = 3,
) -> pd.DataFrame:
    """
    Identify discrete pinch events (onset → offset).

    Parameters
    ----------
    threshold : pinch strength above which a pinch is 'active'
    min_duration_frames : ignore blips shorter than this

    Returns
    -------
    DataFrame with columns: onset_frame, offset_frame, onset_s, offset_s, duration_s, peak_pinch
    """
    active = (df["pinch"] >= threshold).astype(int)
    transitions = active.diff().fillna(0)

    onsets  = df.index[transitions ==  1].tolist()
    offsets = df.index[transitions == -1].tolist()

    # Handle edge case: starts already active
    if active.iloc[0] == 1:
        onsets = [0] + onsets
    # Handle edge case: ends while still active
    if len(onsets) > len(offsets):
        offsets.append(df.index[-1])

    events = []
    for on, off in zip(onsets, offsets):
        duration = off - on
        if duration < min_duration_frames:
            continue
        events.append({
            "onset_frame":  on,
            "offset_frame": off,
            "onset_s":      df.loc[on,  "elapsed_s"],
            "offset_s":     df.loc[off, "elapsed_s"],
            "duration_s":   df.loc[off, "elapsed_s"] - df.loc[on, "elapsed_s"],
            "peak_pinch":   df.loc[on:off, "pinch"].max(),
        })

    return pd.DataFrame(events)


# --- Validation helper ------------------------------------------------------

def validate(df: pd.DataFrame) -> dict:
    """
    Sanity checks on a loaded Leap Motion DataFrame.
    Returns: {"ok": bool, "issues": [str]}
    """
    issues: list[str] = []

    if df.empty:
        return {"ok": False, "issues": ["DataFrame is empty"]}

    # Pinch/grab must be in [0, 1]
    for col in ["pinch", "grab"]:
        if col in df.columns:
            if df[col].min() < -0.01 or df[col].max() > 1.01:
                issues.append(f"{col} out of [0, 1] range: [{df[col].min():.3f}, {df[col].max():.3f}]")

    # Palm positions should be non-zero (sensor tracking)
    if "palm_ty" in df.columns:
        if (df["palm_ty"] == 0).all():
            issues.append("palm_ty is all zero — possible tracking failure")

    # No NaNs in key columns
    key_cols = ["pinch", "grab", "palm_tx", "palm_ty", "palm_tz"]
    for col in key_cols:
        if col in df.columns and df[col].isna().any():
            issues.append(f"{col}: contains NaN values")

    # Elapsed time should be strictly increasing
    if "elapsed_s" in df.columns and not df["elapsed_s"].is_monotonic_increasing:
        issues.append("elapsed_s is not monotonically increasing")

    return {"ok": len(issues) == 0, "issues": issues}


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    path = sys.argv[1] if len(sys.argv) > 1 else "leap_pinch1.csv"
    df = load(path)
    print(df[["frame", "elapsed_s", "pinch", "grab", "palm_tx", "palm_ty", "palm_tz"]].head(10).to_string())
    print("\nShape:", df.shape)
    print("Columns:", list(df.columns))
    events = detect_pinch_events(df)
    print(f"\nPinch events detected: {len(events)}")
    if not events.empty:
        print(events.to_string())
    result = validate(df)
    print("\nValidation:", result)