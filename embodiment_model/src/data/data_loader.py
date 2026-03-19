"""
data_loader.py
--------------
Loads a post-session embodiment recording from disk by calling the three
sensor parsers (bioradio, leap_motion, apple_watch) on the raw source files.

Expected session directory layout
----------------------------------
data/
  raw/
    embodiment/
      <session_id>/                    e.g.  P01_control_trial001/
        <any_name>.bcrx                      bioradio recording
        <any_name>_leap.csv                  leap motion export (must contain "leap")
        <any_name>_watch.xml  (or .zip)      apple health export (optional)
        labels.json                          ground truth + metadata
  processed/
    embodiment/
      <session_id>/
        feature_matrix.csv                   output of dataset_builder (regeneratable)
        session_report.json                  QC report + timing

labels.json schema
-------------------
{
  "participant_id":  "P01",
  "condition":       "control",          // e.g. "control", "vibrotactile", "visual"
  "trial_number":    1,
  "embodiment_score": 72.5,             // 0-100 continuous ground truth
  "veq_ownership":   4,                 // optional Likert items
  "veq_agency":      3,
  "veq_location":    5,
  "session_start":   "2026-01-27T13:33:16+00:00",   // ISO-8601, used to align watch data
  "session_end":     "2026-01-27T13:43:16+00:00",
  "notes":           ""
}
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

# The three parsers built in the previous step
from .parsers import bioradio_parser, leapmotion_parser, apple_watch_parser

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class EmbodimentSession:
    """
    One complete post-session recording.

    Sensor DataFrames are stored as returned by the parsers — no pre-processing.
    dataset_builder.py is responsible for time-alignment and feature extraction.

    Attributes
    ----------
    participant_id  : e.g. "P01"
    condition       : experimental condition label
    trial_number    : integer trial index within a participant
    session_dir     : original source directory (for traceability)

    leap_df         : DataFrame from leapmotion_parser.load()
                      columns: frame, elapsed_s, [timestamp], pinch, grab,
                               palm_tx/ty/tz, palm_rx/ry/rz, vel_x/y/z, ...
    bioradio_df     : DataFrame from bioradio_parser.load()
                      columns: timestamp, elapsed_s, <channel_name> (e.g. SKN µS)
    watch_df        : DataFrame from apple_watch_parser.load()   (None until available)
                      columns: type, source, start_time, end_time, value, unit

    embodiment_score : continuous [0, 100] ground truth
    veq_scores       : dict with keys ownership / agency / location  (optional)
    session_start    : timezone-aware datetime parsed from labels.json
    session_end      : timezone-aware datetime parsed from labels.json
    """
    # Identity
    participant_id: str
    condition:      str
    trial_number:   int
    session_dir:    Path

    # Raw sensor DataFrames (as-loaded from parsers)
    leap_df:      pd.DataFrame
    bioradio_df:  pd.DataFrame
    watch_df:     Optional[pd.DataFrame]

    # Ground truth
    embodiment_score: float
    veq_scores:       Optional[dict] = field(default_factory=dict)

    # Timing
    session_start: Optional[datetime] = None
    session_end:   Optional[datetime] = None

    # Free-text
    notes: str = ""

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def duration_s(self) -> Optional[float]:
        if self.session_start and self.session_end:
            return (self.session_end - self.session_start).total_seconds()
        return None

    @property
    def session_id(self) -> str:
        return f"{self.participant_id}_{self.condition}_trial{self.trial_number:03d}"

    def __repr__(self) -> str:
        watch_status = f"{len(self.watch_df)} records" if self.watch_df is not None else "not loaded"
        return (
            f"EmbodimentSession("
            f"id={self.session_id!r}, "
            f"score={self.embodiment_score:.1f}, "
            f"leap={len(self.leap_df)} frames, "
            f"bioradio={len(self.bioradio_df)} samples, "
            f"watch={watch_status})"
        )


# ---------------------------------------------------------------------------
# File discovery helpers
# ---------------------------------------------------------------------------

def _find_file(session_dir: Path, pattern: str) -> Optional[Path]:
    """Return the first file matching a glob pattern, or None."""
    matches = sorted(session_dir.glob(pattern))
    if not matches:
        return None
    if len(matches) > 1:
        logger.warning(
            "%s: multiple files match '%s', using %s",
            session_dir.name, pattern, matches[0].name,
        )
    return matches[0]


def _require_file(session_dir: Path, pattern: str) -> Path:
    """Like _find_file but raises FileNotFoundError if nothing matches."""
    path = _find_file(session_dir, pattern)
    if path is None:
        raise FileNotFoundError(
            f"No file matching '{pattern}' in {session_dir}. "
            f"Contents: {[f.name for f in session_dir.iterdir()]}"
        )
    return path


# ---------------------------------------------------------------------------
# Single-session loader
# ---------------------------------------------------------------------------

def load_session(
    session_dir: Path | str,
    leap_rate_hz: float = 60.0,
    watch_source_filter: Optional[str] = "Apple Watch",
) -> EmbodimentSession:
    """
    Load one post-session embodiment recording.

    Parameters
    ----------
    session_dir : directory containing the raw sensor files + labels.json
    leap_rate_hz : frame rate to use when synthesising Leap timestamps
        (Leap Motion CSV has no timestamp column; 60 Hz is the hardware default)
    watch_source_filter : filter Apple Health records to this source device.
        Pass None to include records from all sources (iPhone, Watch, apps).

    Returns
    -------
    EmbodimentSession with all DataFrames populated.

    Raises
    ------
    FileNotFoundError  if required files are missing
    ValueError         if a parser rejects a file
    """
    session_dir = Path(session_dir)
    if not session_dir.is_dir():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    logger.info("Loading session: %s", session_dir.name)

    # ------------------------------------------------------------------ #
    # 1. Labels (always required)                                          #
    # ------------------------------------------------------------------ #
    labels_path = _require_file(session_dir, "labels.json")
    with open(labels_path) as f:
        labels = json.load(f)

    session_start = _parse_dt(labels.get("session_start"))
    session_end   = _parse_dt(labels.get("session_end"))

    # ------------------------------------------------------------------ #
    # 2. BioRadio  (.bcrx — always required)                              #
    # ------------------------------------------------------------------ #
    bcrx_path = _require_file(session_dir, "*.bcrx")
    logger.info("  BioRadio: %s", bcrx_path.name)
    bioradio_df = bioradio_parser.load(bcrx_path)

    # ------------------------------------------------------------------ #
    # 3. Leap Motion  (*leap*.csv — always required)                       #
    # ------------------------------------------------------------------ #
    leap_path = _require_file(session_dir, "*leap*.csv")
    logger.info("  Leap Motion: %s", leap_path.name)
    leap_df = leapmotion_parser.load(
        leap_path,
        frame_rate_hz=leap_rate_hz,
        session_start=session_start,    # attaches absolute timestamps if available
    )

    # ------------------------------------------------------------------ #
    # 4. Apple Watch  (.xml or .zip — optional until available)           #
    # ------------------------------------------------------------------ #
    watch_df: Optional[pd.DataFrame] = None
    watch_path = _find_file(session_dir, "*watch*.xml") \
              or _find_file(session_dir, "*watch*.zip") \
              or _find_file(session_dir, "export.xml") \
              or _find_file(session_dir, "export.zip")

    if watch_path:
        logger.info("  Apple Watch: %s", watch_path.name)
        try:
            raw_watch = apple_watch_parser.load(
                watch_path,
                source_filter=watch_source_filter,
            )
            # Clip to session window if we have timing information
            if session_start and session_end and not raw_watch.empty:
                watch_df = apple_watch_parser.extract_session_window(
                    raw_watch,
                    pd.Timestamp(session_start, tz="UTC"),
                    pd.Timestamp(session_end,   tz="UTC"),
                )
            else:
                watch_df = raw_watch
        except Exception as exc:
            logger.warning("  Apple Watch load failed (%s) — continuing without it", exc)
    else:
        logger.info("  Apple Watch: no file found — will be None in session")

    # ------------------------------------------------------------------ #
    # 5. Build session object                                              #
    # ------------------------------------------------------------------ #
    session = EmbodimentSession(
        participant_id=labels["participant_id"],
        condition=labels["condition"],
        trial_number=int(labels["trial_number"]),
        session_dir=session_dir,
        leap_df=leap_df,
        bioradio_df=bioradio_df,
        watch_df=watch_df,
        embodiment_score=float(labels["embodiment_score"]),
        veq_scores={
            "ownership": labels.get("veq_ownership"),
            "agency":    labels.get("veq_agency"),
            "location":  labels.get("veq_location"),
        },
        session_start=session_start,
        session_end=session_end,
        notes=labels.get("notes", ""),
    )

    logger.info("  Loaded: %s", session)
    return session


# ---------------------------------------------------------------------------
# Multi-session loader
# ---------------------------------------------------------------------------

def load_sessions(
    data_dir: Path | str,
    participant_ids: Optional[list[str]] = None,
    conditions:      Optional[list[str]] = None,
    leap_rate_hz:    float = 60.0,
) -> list[EmbodimentSession]:
    """
    Recursively find and load all session directories under data_dir.

    A valid session directory must contain:
      - labels.json
      - at least one .bcrx file
      - at least one *leap*.csv file

    Apple Watch files are optional (session is still loaded without them).

    Parameters
    ----------
    data_dir : root of the embodiment session tree
        e.g. Capstone-Project/data/raw/embodiment/
    participant_ids : if provided, only load sessions for these IDs
    conditions : if provided, only load sessions with these condition labels
    leap_rate_hz : passed through to load_session()

    Returns
    -------
    List of successfully loaded EmbodimentSession objects.
    """
    data_dir = Path(data_dir)
    sessions: list[EmbodimentSession] = []
    skipped = 0

    # Find every directory that has both a labels.json and a .bcrx
    for candidate in sorted(data_dir.rglob("labels.json")):
        session_dir = candidate.parent

        has_bcrx = bool(list(session_dir.glob("*.bcrx")))
        has_leap = bool(list(session_dir.glob("*leap*.csv")))

        if not (has_bcrx and has_leap):
            logger.debug("Skipping %s — missing .bcrx or *leap*.csv", session_dir.name)
            skipped += 1
            continue

        try:
            session = load_session(session_dir, leap_rate_hz=leap_rate_hz)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", session_dir.name, exc)
            skipped += 1
            continue

        # Apply filters
        if participant_ids and session.participant_id not in participant_ids:
            continue
        if conditions and session.condition not in conditions:
            continue

        sessions.append(session)

    logger.info(
        "Loaded %d session(s), skipped %d from %s",
        len(sessions), skipped, data_dir,
    )
    return sessions


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _parse_dt(value: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 string to a timezone-aware datetime, or return None."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        logger.warning("Could not parse datetime: %r", value)
        return None


def validate_session(session: EmbodimentSession) -> dict:
    """
    Run all three parser validators and combine results.

    Returns
    -------
    {"ok": bool, "issues": {"leap": [...], "bioradio": [...], "watch": [...]}}
    """
    results = {
        "leap":     leapmotion_parser.validate(session.leap_df),
        "bioradio": bioradio_parser.validate(session.bioradio_df),
        "watch":    apple_watch_parser.validate(session.watch_df)
                    if session.watch_df is not None
                    else {"ok": True, "issues": ["watch data not loaded"]},
    }
    all_ok = all(r["ok"] for r in results.values())
    return {"ok": all_ok, "results": results}