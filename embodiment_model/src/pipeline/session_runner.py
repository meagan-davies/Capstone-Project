"""
session_runner.py
-----------------
Post-session async pipeline entry point.

After a test session is complete, drop the raw files into the session
directory and run this to get an embodiment score + QC report.

Usage
-----
    # From repo root:
    python -m embodiment_model.pipeline.session_runner \
        --session-dir data/raw/embodiment/P01_control_trial001 \
        --model-path  artifacts/embodiment-model/embodiment_model.pkl \
        --save-processed data/processed/

    # Or call run_session() directly from integration/combined_pipeline.py:
    import asyncio
    from embodiment_model.pipeline.session_runner import run_session
    result = asyncio.run(run_session(session_dir, model_path))

Pipeline stages (all timed)
----------------------------
  ingest          — load .bcrx, *leap*.csv, watch XML concurrently
  validate        — per-parser QC checks
  feature matrix  — align to 1 Hz grid, build feature vector
  inference       — regressor.predict() → embodiment score 0-100
  save            — write processed outputs to disk
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# ── path resolution (mirrors test_pipeline.py) ────────────────────────────
_script_dir = Path(__file__).resolve().parent          # embodiment_model/pipeline/
_src_root   = _script_dir.parent / "src"               # embodiment_model/src/
_repo_root  = _script_dir.parent.parent                # Capstone-Project/
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_src_root))

from data.data_loader    import load_session, validate_session
from data.dataset_builder import build_feature_matrix, save_processed
from models.trainer      import load_model
from pipeline.timing     import timed

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core async runner
# ---------------------------------------------------------------------------

async def run_session(
    session_dir: Path | str,
    model_path:  Optional[Path | str] = None,
    processed_root: Optional[Path | str] = None,
) -> dict:
    """
    Run the full post-session pipeline for one recording.

    Parameters
    ----------
    session_dir    : directory containing .bcrx, *leap*.csv, labels.json
    model_path     : path to a trained .pkl model (optional — skips inference
                     if not provided, useful for pipeline testing)
    processed_root : if set, writes feature_matrix.csv + session_report.json
                     under processed_root/embodiment/<session_id>/

    Returns
    -------
    dict with keys:
        session_id       : str
        embodiment_score : float | None   (None if no model provided)
        qc               : dict           (coverage report per sensor)
        processed_dir    : str | None     (path written to, if any)
        timing           : dict           (wall-clock seconds per stage)
    """
    import time
    session_dir = Path(session_dir)
    timings: dict[str, float] = {}

    # ── Stage 1: ingest ────────────────────────────────────────────────
    # load_session() already calls all three parsers internally.
    # We wrap it in asyncio.to_thread so it's non-blocking if called
    # alongside other coroutines (e.g. from combined_pipeline.py).
    with timed("ingest"):
        t0 = time.perf_counter()
        session = await asyncio.to_thread(load_session, session_dir)
        timings["ingest"] = time.perf_counter() - t0

    logger.info("Loaded: %s", session)

    # ── Stage 2: validate ──────────────────────────────────────────────
    with timed("validate"):
        t0 = time.perf_counter()
        validation = validate_session(session)
        timings["validate"] = time.perf_counter() - t0

    if not validation["ok"]:
        failing = {
            src: res["issues"]
            for src, res in validation["results"].items()
            if not res["ok"] and "not loaded" not in str(res["issues"])
        }
        if failing:
            logger.warning("Validation issues: %s", failing)

    # ── Stage 3: feature matrix ────────────────────────────────────────
    with timed("feature matrix"):
        t0 = time.perf_counter()
        ds = await asyncio.to_thread(build_feature_matrix, session)
        timings["feature_matrix"] = time.perf_counter() - t0

    logger.info("Feature matrix: %s", ds)

    # ── Stage 4: inference (optional) ─────────────────────────────────
    embodiment_score: Optional[float] = None

    if model_path is not None:
        model_path = Path(model_path)
        if not model_path.exists():
            logger.warning("Model file not found: %s — skipping inference", model_path)
        else:
            with timed("inference"):
                t0 = time.perf_counter()

                model, _cv, _meta = await asyncio.to_thread(load_model, model_path)

                # build_training_dataset summarises to one row per session;
                # here we already have one session so just take the mean vector
                feature_cols = [c for c in ds.features.columns if c != "elapsed_s"]
                X = ds.features[feature_cols].mean().values.reshape(1, -1)

                predictions = model.predict(X)
                embodiment_score = float(predictions[0])
                timings["inference"] = time.perf_counter() - t0

            logger.info("Embodiment score: %.1f", embodiment_score)

    # ── Stage 5: save processed outputs ───────────────────────────────
    processed_dir: Optional[str] = None

    if processed_root is not None:
        with timed("save"):
            t0 = time.perf_counter()
            out_dir = await asyncio.to_thread(save_processed, ds, processed_root)
            processed_dir = str(out_dir)
            timings["save"] = time.perf_counter() - t0

    return {
        "session_id":       session.session_id,
        "embodiment_score": embodiment_score,
        "qc":               ds.qc,
        "processed_dir":    processed_dir,
        "timing":           timings,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Run the post-session embodiment pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--session-dir", type=Path, required=True,
        help="Session directory  (e.g. data/raw/embodiment/P01_control_trial001)",
    )
    parser.add_argument(
        "--model-path", type=Path, default=None,
        help="Trained model .pkl  (e.g. artifacts/embodiment-model/embodiment_model.pkl). "
             "If omitted the pipeline runs without inference — useful for testing.",
    )
    parser.add_argument(
        "--save-processed", type=Path, default=None, metavar="DIR",
        help="Write feature_matrix.csv + session_report.json here  (e.g. data/processed/)",
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s  %(levelname)s  [%(name)s]  %(message)s",
    )

    result = asyncio.run(run_session(
        args.session_dir,
        model_path=args.model_path,
        processed_root=args.save_processed,
    ))

    print("\n" + "=" * 60)
    print(f"  Session:          {result['session_id']}")
    score = result["embodiment_score"]
    print(f"  Embodiment score: {f'{score:.1f}' if score is not None else 'n/a (no model)'}")
    print(f"\n  Sensor coverage:")
    qc = result["qc"]
    print(f"    BioRadio (EDA):  {qc['bioradio_coverage']}")
    print(f"    Leap Motion:     {qc['leap_coverage']}")
    print(f"    Apple Watch:     {qc['watch_coverage']}")
    print(f"\n  Timing (seconds):")
    for stage, secs in result["timing"].items():
        print(f"    {stage:<20} {secs:.3f}s")
    if result["processed_dir"]:
        print(f"\n  Saved to: {result['processed_dir']}")
    print("=" * 60 + "\n")

    # Also write a machine-readable result to stdout for piping
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()