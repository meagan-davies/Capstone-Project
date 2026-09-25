"""
data_loader.py
--------------
Loads one embodiment trial/session from disk using the BioRadio, Leap Motion,
and optional Apple Watch parsers.

Expected session directory layout
----------------------------------
data/
  raw/
    embodiment/
      <session_id>/
        <any_name>.bcrx
        <any_name>_leap.csv
        <any_name>_watch.xml  (optional)
        labels.json

labels.json schema
------------------
{
  "participant_id": "T01",
  "condition": "post",             // pre, pros, or post
  "trial_number": 1,               // 1=Grasp, 2=Zip, 3=Block
  "embodiment_score": 100,         // task-level freeze-check score
  "session_start": "2026-04-08T13:52:11+00:00",
  "session_end": "2026-04-08T13:52:33+00:00",
  "questionnaire": 47,             // post-test questionnaire score
  "notes": ""
}

The questionnaire score is retained as metadata and is NOT used as a
predictor in the embodiment model.

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

CONDITION_LABELS = {
    "pre": "Biological Before",
    "pros": "Prosthetic",
    "post": "Biological After",
}

TASK_LABELS = {
    1: "Grasp",
    2: "Zip",
    3: "Block",
}

# ---------------------------------------------------------------------------
# Session dataclass
# ---------------------------------------------------------------------------

@dataclass
class EmbodimentSession:
    """
    One complete embodiment trial/session.

    Each session corresponds to one participant, condition, and task.
    The embodiment_score is the freeze-check score collected for that
    specific task.

    Attributes
    ----------
    participant_id (str): Unique participant identifier, e.g. ``"P08"``.

    condition (str): Experimental condition. Expected values are:
        ``"pre"`` for Biological Before,
        ``"pros"`` for Prosthetic, and
        ``"post"`` for Biological After.

    trial_number (int): Task identifier within the experiment:
        ``1`` = Grasp, ``2`` = Zip, ``3`` = Block.

    session_dir (Path): Path to the raw session directory. Retained for traceability and
        access to the original source files.

    leap_df (DataFrame): Parsed Leap Motion recording. Contains frame-level hand and movement
        measurements such as pinch strength, grab strength, palm position,
        orientation, and velocity.

    bioradio_df (DataFrame): Parsed BioRadio recording. Contains timestamped physiological
        measurements, including skin conductance / electrodermal activity.

    watch_df (DataFrame): Optional parsed Apple Watch recording. Contains timestamped Apple
        Health measurements when an Apple Watch export is available.
        ``None`` when no Apple Watch data were collected.

    freeze_check_score (float): Task-level embodiment score obtained from the freeze-check questions
        administered after the task. This is the ground-truth target used
        for machine-learning prediction.

    questionnaire_score (float): Score from the post-test embodiment questionnaire. This is retained
        as metadata for analysis and reporting and is not used as a model
        input feature.

    session_start (datetime): Time at which the recording session began. Expected to be a
        timezone-aware datetime parsed from the ISO-8601 value in ``labels.json``.

    session_end (datetime): Time at which the recording session ended. Expected to be a
        timezone-aware datetime parsed from the ISO-8601 value in ``labels.json``.

    notes (str): Optional free-text notes associated with the session
    """

    # Identity
    participant_id: str
    condition: str
    trial_number: int
    session_dir: Path

    # Raw sensor DataFrames
    leap_df: pd.DataFrame
    bioradio_df: pd.DataFrame
    watch_df: Optional[pd.DataFrame]

    # Ground truth / metadata
    freeze_check_score: float
    questionnaire_score: Optional[float] = None

    # Timing
    session_start: Optional[datetime] = None
    session_end: Optional[datetime] = None

    # Free-text
    notes: str = ""

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def condition_label(self) -> str:
        return CONDITION_LABELS.get(self.condition, self.condition)

    @property
    def task_label(self) -> str:
        return TASK_LABELS.get(self.trial_number, f"Unknown ({self.trial_number})")

    @property
    def duration_s(self) -> Optional[float]:
        if self.session_start and self.session_end:
            return (self.session_end - self.session_start).total_seconds()
        return None

    @property
    def session_id(self) -> str:
        return f"{self.participant_id}_{self.condition}_trial{self.trial_number:03d}"

    def __repr__(self) -> str:
        watch_status = (
            f"{len(self.watch_df)} records"
            if self.watch_df is not None
            else "not loaded"
        )

        questionnaire = (
            f"{self.questionnaire_score:.1f}"
            if self.questionnaire_score is not None
            else "None"
        )

        return (
            f"EmbodimentSession("
            f"id={self.session_id!r}, "
            f"condition={self.condition_label!r}, "
            f"task={self.task_label!r}, "
            f"freeze_check={self.freeze_check_score:.1f}, "
            f"questionnaire={questionnaire}, "
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

    VALID_CONDITIONS = {"pre", "pros", "post"}
    VALID_TASKS = {1, 2, 3}

    condition = labels["condition"].lower()

    if condition not in VALID_CONDITIONS:
        raise ValueError(
            f"Invalid condition {condition!r} in {labels_path}. "
            f"Expected one of {sorted(VALID_CONDITIONS)}."
        )

    trial_number = int(labels["trial_number"])

    if trial_number not in VALID_TASKS:
        raise ValueError(
            f"Invalid trial_number {trial_number} in {labels_path}. "
            f"Expected one of {sorted(VALID_TASKS)}."
        )

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
        freeze_check_score=float(labels["embodiment_score"]),
        questionnaire_score=(
            float(labels["questionnaire"])
            if labels.get("questionnaire") is not None
            else None
        ),
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
    """Parse an ISO-8601 datetime, treating labels as Calgary local time."""
    if not value:
        return None

    try:
        dt = datetime.fromisoformat(value)
        # labels.json stores Calgary clock time but may be marked +00:00.
        # Reinterpret the clock time as Mountain Time rather than converting it.
        if dt.tzinfo is not None:
            from zoneinfo import ZoneInfo
            dt = dt.replace(tzinfo=ZoneInfo("America/Edmonton"))

        return dt

    except ValueError:
        logger.warning("Could not parse datetime: %r", value)
        return None


def validate_session(session: EmbodimentSession) -> dict:
    """
    Run validators for all available sensor sources.

    Apple Watch validation is performed only when Watch data are present.
    """

    results = {
        "leap": leapmotion_parser.validate(session.leap_df),
        "bioradio": bioradio_parser.validate(session.bioradio_df),
    }

    if session.watch_df is not None:
        results["watch"] = apple_watch_parser.validate(session.watch_df)
    else:
        results["watch"] = {
            "ok": True,
            "issues": [],
            "status": "not_available",
        }

    all_ok = all(r["ok"] for r in results.values())

    return {
        "ok": all_ok,
        "results": results,
    }