"""
dataset_builder.py
------------------
Takes the raw parser DataFrames stored in EmbodimentSession and builds
a single aligned feature matrix ready for the embodiment regressor.

Pipeline
--------
  EmbodimentSession
      │
      ├─ bioradio_df   (500 Hz,   absolute timestamps)
      ├─ leap_df       (60 Hz,    elapsed_s + optional absolute timestamps)
      └─ watch_df      (sparse,   absolute timestamps — HRV every ~5 min, HR every ~5 s)
      │
      ▼
  [1] Normalise all sources to a common UTC time index
  [2] Resample every source to the target grid  (default: 1 Hz)
  [3] Stack into one wide DataFrame (one row per second)
  [4] Impute short gaps (forward-fill ≤ max_gap_s)
  [5] Return feature matrix + a QC report

Columns in the output feature matrix
--------------------------------------
  elapsed_s           — seconds since session start (float)
  eda_mean            — mean skin conductance in window (µS)
  eda_std             — std  skin conductance in window (µS)
  pinch_mean          — mean Leap pinch strength in window [0,1]
  pinch_max           — max  Leap pinch strength in window
  grab_mean           — mean Leap grab strength in window [0,1]
  palm_speed_mean     — mean hand speed in window (mm/s)
  palm_speed_max      — max  hand speed in window (mm/s)
  pinch_events        — number of discrete pinch events in window
  hrv_ms              — HRV SDNN (ms)  [from watch, sparse — ffill applied]
  heart_rate_bpm      — heart rate (bpm) [from watch, ffill applied]
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

try:
    from .data_loader import EmbodimentSession
    from .parsers import leapmotion_parser
except ImportError:
    from data.data_loader import EmbodimentSession   # fallback for direct script runs
    from data.parsers import leapmotion_parser

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DEFAULT_RESAMPLE_HZ   = 1          # target grid: 1 row per second
DEFAULT_WINDOW_S      = 1.0        # aggregation window per row (same as 1/resample)
DEFAULT_MAX_GAP_S     = 30         # forward-fill watch data up to this many seconds
WATCH_RESAMPLE_RULE   = "1s"       # pandas offset alias for watch resampling

# Watch metric columns we want to carry forward (must match apple_watch_parser type names)
WATCH_METRICS = ["hrv_ms", "heart_rate_bpm"]


# ---------------------------------------------------------------------------
# Output container
# ---------------------------------------------------------------------------

@dataclass
class AlignedDataset:
    """
    Output of build_feature_matrix().

    Attributes
    ----------
    features    : wide DataFrame, one row per second, ready for the regressor
    qc          : quality-control report (gaps, coverage per source)
    session_id  : from EmbodimentSession.session_id
    label       : embodiment_score  (target variable)
    veq_scores  : raw VEQ questionnaire responses (optional)
    """
    features:   pd.DataFrame
    qc:         dict
    session_id: str
    label:      float
    veq_scores: Optional[dict] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"AlignedDataset(session={self.session_id!r}, "
            f"rows={len(self.features)}, "
            f"cols={list(self.features.columns)}, "
            f"label={self.label:.1f})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _bioradio_to_grid(
    bioradio_df: pd.DataFrame,
    grid: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Resample BioRadio data onto the common UTC grid.

    BioRadio gives us:
        timestamp (UTC, absolute), elapsed_s, <channel> (e.g. SKN in µS)

    We compute mean and std per grid cell (1-second window) as EDA features.
    """
    # Find the EDA column — any non-index column that isn't timestamp/elapsed_s
    signal_cols = [c for c in bioradio_df.columns if c not in ("timestamp", "elapsed_s")]
    if not signal_cols:
        raise ValueError("bioradio_df has no signal columns after timestamp/elapsed_s")

    eda_col = signal_cols[0]   # SKN = skin conductance
    ts = bioradio_df.set_index("timestamp")[eda_col]
    ts.index = pd.to_datetime(ts.index, utc=True)

    # Resample to 1-second windows
    rs = ts.resample(WATCH_RESAMPLE_RULE)
    bio_grid = pd.DataFrame({
        "eda_mean": rs.mean(),
        "eda_std":  rs.std().fillna(0),
    })

    # Reindex onto the common grid
    bio_grid = bio_grid.reindex(grid, method="nearest", tolerance="1s")
    return bio_grid


def _leap_to_grid(
    leap_df: pd.DataFrame,
    grid: pd.DatetimeIndex,
    session_start: pd.Timestamp,
) -> pd.DataFrame:
    """
    Aggregate Leap Motion data onto the common UTC grid.

    Leap gives us:
        frame, elapsed_s, [timestamp], pinch, grab, palm_tx/ty/tz, vel_x/y/z, ...

    Computes per-second:
        pinch_mean, pinch_max, grab_mean,
        palm_speed_mean, palm_speed_max,
        pinch_events  (count of discrete pinch onsets)
    """
    # Synthesise absolute timestamps if not already present
    if "timestamp" not in leap_df.columns:
        leap_df = leap_df.copy()
        leap_df["timestamp"] = session_start + pd.to_timedelta(leap_df["elapsed_s"], unit="s")

    leap_df = leap_df.set_index(pd.to_datetime(leap_df["timestamp"], utc=True))

    # Palm speed (mm/s)
    speed = leapmotion_parser.compute_velocity_magnitude(leap_df)
    leap_df = leap_df.copy()
    leap_df["palm_speed"] = speed

    rs = leap_df.resample(WATCH_RESAMPLE_RULE)

    leap_grid = pd.DataFrame({
        "pinch_mean":       rs["pinch"].mean(),
        "pinch_max":        rs["pinch"].max(),
        "grab_mean":        rs["grab"].mean(),
        "palm_speed_mean":  rs["palm_speed"].mean(),
        "palm_speed_max":   rs["palm_speed"].max(),
    })

    # Pinch event count per second
    events = leapmotion_parser.detect_pinch_events(
        leap_df.reset_index(drop=True).assign(
            elapsed_s=(leap_df.index - leap_df.index[0]).total_seconds()
        )
    )
    if not events.empty:
        onset_ts = session_start + pd.to_timedelta(events["onset_s"], unit="s")
        onset_ts = pd.to_datetime(onset_ts, utc=True)
        event_counts = onset_ts.dt.floor("1s").value_counts().rename("pinch_events")
        leap_grid = leap_grid.join(event_counts, how="left")
    else:
        leap_grid["pinch_events"] = 0

    leap_grid["pinch_events"] = leap_grid["pinch_events"].fillna(0).astype(int)
    leap_grid = leap_grid.reindex(grid, method="nearest", tolerance="1s")
    return leap_grid


def _watch_to_grid(
    watch_df: Optional[pd.DataFrame],
    grid: pd.DatetimeIndex,
    max_gap_s: int,
) -> pd.DataFrame:
    """
    Resample sparse Apple Watch metrics onto the common UTC grid.

    Watch data is very sparse (HRV every ~5 min, HR every few seconds).
    We forward-fill up to max_gap_s after each reading.
    """
    empty = pd.DataFrame(
        {m: np.nan for m in WATCH_METRICS},
        index=grid,
    )

    if watch_df is None or watch_df.empty:
        logger.debug("watch_df is None/empty — watch columns will be NaN")
        return empty

    frames = []
    for metric in WATCH_METRICS:
        sub = watch_df[watch_df["type"] == metric].copy()
        if sub.empty:
            frames.append(pd.Series(np.nan, index=grid, name=metric))
            continue

        sub = sub.set_index("start_time")["value"]
        sub.index = pd.to_datetime(sub.index, utc=True)
        resampled = sub.resample(WATCH_RESAMPLE_RULE).mean()
        # Forward-fill (sparse sensor — reading lasts until the next one)
        resampled = resampled.reindex(grid).ffill(limit=max_gap_s)
        frames.append(resampled.rename(metric))

    return pd.concat(frames, axis=1)


def _build_common_grid(
    session_start: pd.Timestamp,
    session_end:   pd.Timestamp,
    resample_hz:   int,
) -> pd.DatetimeIndex:
    """Build a UTC DatetimeIndex from session_start to session_end at resample_hz."""
    freq = pd.tseries.frequencies.to_offset(f"{1000 // resample_hz}ms")
    return pd.date_range(
        start=session_start.floor("1s"),
        end=session_end.ceil("1s"),
        freq=freq,
        tz="UTC",
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_feature_matrix(
    session: EmbodimentSession,
    resample_hz:  int = DEFAULT_RESAMPLE_HZ,
    max_gap_s:    int = DEFAULT_MAX_GAP_S,
) -> AlignedDataset:
    """
    Align all three sensor sources and produce a feature matrix.

    Parameters
    ----------
    session     : loaded EmbodimentSession from data_loader
    resample_hz : grid frequency in Hz (1 = one row per second)
    max_gap_s   : maximum seconds to forward-fill sparse watch readings

    Returns
    -------
    AlignedDataset with .features DataFrame and .qc report.

    Raises
    ------
    ValueError  if session_start / session_end are missing from labels.json
                (they are needed to establish the common time axis)
    """
    # ------------------------------------------------------------------ #
    # 0. Establish common time axis                                        #
    # ------------------------------------------------------------------ #
    if not (session.session_start and session.session_end):
        raise ValueError(
            f"Session {session.session_id} is missing session_start / session_end "
            f"in labels.json — these are required to build the common time axis."
        )

    t_start = pd.Timestamp(session.session_start).tz_convert("UTC") \
              if session.session_start.tzinfo else \
              pd.Timestamp(session.session_start, tz="UTC")
    t_end   = pd.Timestamp(session.session_end).tz_convert("UTC") \
              if session.session_end.tzinfo else \
              pd.Timestamp(session.session_end, tz="UTC")
    grid    = _build_common_grid(t_start, t_end, resample_hz)

    logger.info(
        "Building feature matrix for %s | grid: %d rows @ %d Hz",
        session.session_id, len(grid), resample_hz,
    )

    # ------------------------------------------------------------------ #
    # 1. BioRadio → EDA features                                          #
    # ------------------------------------------------------------------ #
    bio_features = _bioradio_to_grid(session.bioradio_df, grid)

    # ------------------------------------------------------------------ #
    # 2. Leap Motion → control features                                   #
    # ------------------------------------------------------------------ #
    leap_features = _leap_to_grid(session.leap_df, grid, t_start)

    # ------------------------------------------------------------------ #
    # 3. Apple Watch → HRV / HR (sparse, forward-filled)                  #
    # ------------------------------------------------------------------ #
    watch_features = _watch_to_grid(session.watch_df, grid, max_gap_s)

    # ------------------------------------------------------------------ #
    # 4. Concatenate into one wide frame                                   #
    # ------------------------------------------------------------------ #
    features = pd.concat([bio_features, leap_features, watch_features], axis=1)

    # Attach elapsed_s for interpretability / plotting
    features.insert(0, "elapsed_s",
                    (features.index - t_start).total_seconds())

    # ------------------------------------------------------------------ #
    # 5. QC report                                                         #
    # ------------------------------------------------------------------ #
    qc = _build_qc(features, session)

    logger.info(
        "Feature matrix: %d rows × %d cols | NaN coverage: %s",
        len(features), len(features.columns),
        {c: f"{features[c].isna().mean():.1%}" for c in features.columns if features[c].isna().any()},
    )

    return AlignedDataset(
        features=features,
        qc=qc,
        session_id=session.session_id,
        label=session.embodiment_score,
        veq_scores=session.veq_scores,
    )


def build_training_dataset(
    sessions: list[EmbodimentSession],
    resample_hz: int = DEFAULT_RESAMPLE_HZ,
    max_gap_s:   int = DEFAULT_MAX_GAP_S,
    drop_na_threshold: float = 0.5,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Build X (features) and y (labels) from a list of sessions for model training.

    Each session produces one row in X (mean across the session window) and
    one value in y (embodiment_score).

    Parameters
    ----------
    sessions           : list from data_loader.load_sessions()
    resample_hz        : passed to build_feature_matrix()
    max_gap_s          : passed to build_feature_matrix()
    drop_na_threshold  : drop feature columns where NaN fraction > this value

    Returns
    -------
    X : DataFrame of shape (n_sessions, n_features)
    y : Series of shape (n_sessions,) — embodiment scores
    """
    rows   : list[pd.Series] = []
    labels : list[float]     = []
    ids    : list[str]        = []

    for session in sessions:
        try:
            ds = build_feature_matrix(session, resample_hz, max_gap_s)
        except Exception as exc:
            logger.warning("Skipping %s: %s", session.session_id, exc)
            continue

        # Summarise to one vector per session (mean of all rows)
        feature_cols = [c for c in ds.features.columns if c != "elapsed_s"]
        row = ds.features[feature_cols].mean()
        rows.append(row)
        labels.append(ds.label)
        ids.append(ds.session_id)

    if not rows:
        raise ValueError("No sessions could be processed — check logs for errors")

    X = pd.DataFrame(rows, index=ids)
    y = pd.Series(labels, index=ids, name="embodiment_score")

    # Drop columns with too many NaNs (e.g. watch data not yet available)
    nan_fracs = X.isna().mean()
    drop_cols = nan_fracs[nan_fracs > drop_na_threshold].index.tolist()
    if drop_cols:
        logger.info("Dropping %d high-NaN column(s): %s", len(drop_cols), drop_cols)
        X = X.drop(columns=drop_cols)

    logger.info("Training dataset: X=%s, y=%s", X.shape, y.shape)
    return X, y


# ---------------------------------------------------------------------------
# QC helper
# ---------------------------------------------------------------------------

def _build_qc(features: pd.DataFrame, session: EmbodimentSession) -> dict:
    """Build a per-source quality report."""
    n = len(features)

    def coverage(cols: list[str]) -> str:
        sub = features[[c for c in cols if c in features.columns]]
        if sub.empty:
            return "0.0%"
        return f"{(1 - sub.isna().mean().mean()):.1%}"

    return {
        "session_id":       session.session_id,
        "duration_s":       session.duration_s,
        "grid_rows":        n,
        "bioradio_coverage": coverage(["eda_mean", "eda_std"]),
        "leap_coverage":    coverage(["pinch_mean", "grab_mean", "palm_speed_mean"]),
        "watch_coverage":   coverage(WATCH_METRICS),
        "nan_by_column":    {c: float(features[c].isna().mean())
                             for c in features.columns
                             if features[c].isna().any()},
        "watch_loaded":     session.watch_df is not None,
    }


# ---------------------------------------------------------------------------
# Processed output writer
# ---------------------------------------------------------------------------

def save_processed(
    ds: AlignedDataset,
    processed_root: Path | str,
) -> Path:
    """
    Write AlignedDataset outputs to the processed/ directory tree.

    Creates:
        processed/embodiment/<session_id>/feature_matrix.csv
        processed/embodiment/<session_id>/session_report.json

    Parameters
    ----------
    ds             : output of build_feature_matrix()
    processed_root : path to the processed/ folder
                     e.g. Capstone-Project/data/processed/

    Returns
    -------
    Path to the session's processed directory.
    """
    import json

    out_dir = Path(processed_root) / "embodiment" / ds.session_id
    out_dir.mkdir(parents=True, exist_ok=True)

    # Feature matrix — keep timestamp index for traceability
    feat_path = out_dir / "feature_matrix.csv"
    ds.features.to_csv(feat_path)
    logger.info("Saved feature matrix → %s", feat_path)

    # Session report — QC + label + VEQ
    report = {
        **ds.qc,
        "label":      ds.label,
        "veq_scores": ds.veq_scores,
    }
    report_path = out_dir / "session_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info("Saved session report → %s", report_path)

    return out_dir