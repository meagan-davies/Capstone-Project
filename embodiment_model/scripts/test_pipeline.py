"""
test_pipeline.py
----------------
Verify the full data loading + feature building pipeline works with your
real session files, and optionally write processed outputs to disk.

Usage (from repo root)
-----------------------
  # Test a single session
  python embodiment_model/scripts/test_pipeline.py \
      --session-dir data/raw/embodiment/P01_control_trial001

  # Test every session in the folder
  python embodiment_model/scripts/test_pipeline.py \
      --data-dir data/raw/embodiment/

  # Test + write feature_matrix.csv and session_report.json
  python embodiment_model/scripts/test_pipeline.py \
      --data-dir data/raw/embodiment/ \
      --save-processed data/processed/

  # Verbose: also prints the feature matrix row-by-row
  python embodiment_model/scripts/test_pipeline.py \
      --session-dir data/raw/embodiment/P01_control_trial001 --verbose

What it checks
--------------
  [1] File discovery — *.bcrx, *leap*.csv, optional watch, labels.json
  [2] Session load   — data_loader assembles EmbodimentSession correctly
  [3] Validation     — all three parser validators pass
  [4] Alignment      — dataset_builder produces a feature matrix
  [5] QC report      — coverage per sensor source, NaN flagging
  [6] Save processed — (only with --save-processed) writes outputs to disk

Repo layout this script expects
---------------------------------
  data/
    raw/
      embodiment/
        P01_control_trial001/     ← one folder per session
          *.bcrx                  BioRadio recording  (required)
          *leap*.csv              Leap Motion export  (required)
          *watch*.xml or .zip     Apple Health export (optional)
          labels.json             Ground truth        (required)
    processed/                    ← created automatically by --save-processed
      embodiment/
        P01_control_trial001/
          feature_matrix.csv
          session_report.json

Minimal labels.json
--------------------
{
  "participant_id":   "P01",
  "condition":        "control",
  "trial_number":     1,
  "embodiment_score": 0.0,
  "session_start":    "2026-01-27T20:33:16+00:00",
  "session_end":      "2026-01-27T20:33:25+00:00",
  "notes":            "DUMMY — pipeline test, no real questionnaire data yet"
}
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

# Path resolution: insert src/ at index 0 so our `data` package always
# wins over any venv package that happens to share the name `data`.
_script_dir = Path(__file__).resolve().parent          # embodiment-model/scripts/
_src_root   = _script_dir.parent / "src"               # embodiment-model/src/
_repo_root  = _script_dir.parent.parent                # Capstone-Project/
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_src_root))  # index 0: wins over any venv `data` package

from data.data_loader    import load_session, load_sessions, validate_session
from data.dataset_builder import build_feature_matrix, save_processed

# ── formatting helpers ─────────────────────────────────────────────────────

OK   = "✓"
WARN = "⚠"
FAIL = "✗"

def _header(text: str):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}")

def _check(label: str, passed: bool, detail: str = "") -> bool:
    icon = OK if passed else FAIL
    line = f"  {icon}  {label}"
    if detail:
        line += f"  — {detail}"
    print(line)
    return passed


# ── single-session test ────────────────────────────────────────────────────

def test_session(
    session_dir: Path,
    verbose: bool = False,
    processed_root: Optional[Path] = None,
) -> bool:
    """
    Run all checks for one session directory.
    Returns True if required checks pass (missing watch file is a warning, not a failure).
    """
    _header(f"Testing: {session_dir.name}")
    all_ok = True
    t0 = time.perf_counter()

    # ── [1] File discovery ───────────────────────────────────────────────
    print("\n[1] File discovery")
    has_bcrx   = bool(list(session_dir.glob("*.bcrx")))
    has_leap   = bool(list(session_dir.glob("*leap*.csv")))
    has_watch  = bool(
        list(session_dir.glob("*watch*.xml")) +
        list(session_dir.glob("*watch*.zip")) +
        list(session_dir.glob("export.xml"))  +
        list(session_dir.glob("export.zip"))
    )
    has_labels = (session_dir / "labels.json").exists()

    all_ok &= _check("labels.json",  has_labels,
                     "" if has_labels else "REQUIRED — see docstring for minimal template")
    all_ok &= _check("*.bcrx",       has_bcrx,
                     "" if has_bcrx  else "REQUIRED — place BioRadio file here")
    all_ok &= _check("*leap*.csv",   has_leap,
                     "" if has_leap  else 'REQUIRED — filename must contain "leap"')
    _check("watch file", has_watch,
           "found" if has_watch else "(optional — not found, will be None)")

    if not (has_labels and has_bcrx and has_leap):
        print(f"\n  {FAIL}  Missing required files — skipping remaining checks")
        return False

    # ── [2] Session load ─────────────────────────────────────────────────
    print("\n[2] Loading session (data_loader)")
    try:
        t_load = time.perf_counter()
        session = load_session(session_dir)
        load_time = time.perf_counter() - t_load

        _check("load_session()",  True,
               f"{load_time:.2f}s")
        _check("BioRadio loaded", not session.bioradio_df.empty,
               f"{len(session.bioradio_df)} samples  cols={list(session.bioradio_df.columns)}")
        _check("Leap loaded",     not session.leap_df.empty,
               f"{len(session.leap_df)} frames  "
               f"pinch [{session.leap_df['pinch'].min():.2f}, {session.leap_df['pinch'].max():.2f}]")
        _check("Watch loaded",    session.watch_df is not None,
               f"{len(session.watch_df)} records" if session.watch_df is not None
               else "None (add Apple Watch export when available)")
        _check("Embodiment score", 0 <= session.embodiment_score <= 100,
               f"{session.embodiment_score}  (0.0 = dummy placeholder)")
        _check("Session timing",  bool(session.session_start and session.session_end),
               f"{session.duration_s}s" if session.duration_s
               else "MISSING — add session_start/end to labels.json")

    except Exception as exc:
        _check("load_session()", False, str(exc))
        return False

    # ── [3] Parser validation ────────────────────────────────────────────
    print("\n[3] Parser validation")
    v = validate_session(session)
    for src, result in v["results"].items():
        if result["issues"]:
            for issue in result["issues"]:
                is_warn = "not loaded" in issue
                print(f"  {WARN if is_warn else FAIL}  {src}: {issue}")
                if not is_warn:
                    all_ok = False
        else:
            _check(src, True)

    # ── [4] Feature matrix ───────────────────────────────────────────────
    print("\n[4] Building feature matrix (dataset_builder)")

    if not (session.session_start and session.session_end):
        print(f"  {WARN}  Skipping — session_start/end missing from labels.json")
        return all_ok

    try:
        t_feat = time.perf_counter()
        ds     = build_feature_matrix(session)
        feat_time = time.perf_counter() - t_feat

        _check("build_feature_matrix()", True,
               f"{feat_time:.2f}s")
        _check("Feature matrix shape",   len(ds.features) > 0,
               f"{ds.features.shape[0]} rows × {ds.features.shape[1]} cols")
        _check("Label preserved",        ds.label == session.embodiment_score,
               f"{ds.label}")

    except Exception as exc:
        _check("build_feature_matrix()", False, str(exc))
        return False

    # ── [5] QC coverage report ───────────────────────────────────────────
    print("\n[5] QC coverage report")
    qc = ds.qc
    _check("BioRadio (EDA) coverage", True, qc["bioradio_coverage"])
    _check("Leap coverage",           True, qc["leap_coverage"])
    _check("Watch (HRV/HR) coverage", True,
           qc["watch_coverage"] +
           (" — NaN until Apple Watch export added" if not qc["watch_loaded"] else ""))

    nan_cols = qc.get("nan_by_column", {})
    if nan_cols:
        print(f"\n  {WARN}  Columns with NaN (expected for now):")
        for col, frac in nan_cols.items():
            print(f"       {col}: {frac:.0%} NaN")

    if verbose:
        print("\n  Feature matrix:")
        print(ds.features.to_string())

    # ── [6] Save processed outputs ───────────────────────────────────────
    if processed_root is not None:
        print("\n[6] Saving processed outputs")
        try:
            out_dir = save_processed(ds, processed_root)
            _check("feature_matrix.csv",   (out_dir / "feature_matrix.csv").exists(),
                   str(out_dir / "feature_matrix.csv"))
            _check("session_report.json",  (out_dir / "session_report.json").exists(),
                   str(out_dir / "session_report.json"))
        except Exception as exc:
            _check("save_processed()", False, str(exc))
            all_ok = False

    total  = time.perf_counter() - t0
    status = OK if all_ok else FAIL
    print(f"\n  {status}  {'PASSED' if all_ok else 'FAILED'}  ({total:.2f}s total)")
    return all_ok


# ── multi-session runner ───────────────────────────────────────────────────

def test_data_dir(
    data_dir: Path,
    verbose: bool = False,
    processed_root: Optional[Path] = None,
):
    _header(f"Scanning: {data_dir}")

    label_files = sorted(data_dir.rglob("labels.json"))
    if not label_files:
        print(f"  {FAIL}  No labels.json found under {data_dir}")
        print(       "       Each session needs its own subfolder with a labels.json")
        print(f"\n  Expected layout:")
        print(f"    {data_dir}/")
        print(f"      P01_control_trial001/")
        print(f"        bioradio1.bcrx")
        print(f"        leap_pinch1.csv")
        print(f"        labels.json")
        return

    print(f"  Found {len(label_files)} session(s)\n")
    results: dict[str, bool] = {}
    for lf in label_files:
        sdir = lf.parent
        results[sdir.name] = test_session(
            sdir, verbose=verbose, processed_root=processed_root,
        )

    _header("Summary")
    for name, ok in results.items():
        print(f"  {OK if ok else FAIL}  {name}")
    passed = sum(results.values())
    total  = len(results)
    print(f"\n  {passed}/{total} session(s) passed")
    if processed_root:
        print(f"\n  Processed outputs written to: {processed_root / 'embodiment'}/")


# ── entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test the embodiment model data pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--session-dir", type=Path, metavar="DIR",
        help="Path to a single session directory  "
             "(e.g. data/raw/embodiment/P01_control_trial001)",
    )
    group.add_argument(
        "--data-dir", type=Path, metavar="DIR",
        help="Root directory to scan recursively  "
             "(e.g. data/raw/embodiment/)",
    )

    parser.add_argument(
        "--save-processed", type=Path, metavar="DIR", default=None,
        help="Write feature_matrix.csv + session_report.json here  "
             "(e.g. data/processed/).  Subdirectory embodiment/<session_id>/ "
             "is created automatically.",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Print the full feature matrix row-by-row",
    )
    parser.add_argument(
        "--log-level", default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Internal logging verbosity (default: WARNING)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s [%(name)s]: %(message)s",
    )

    if args.session_dir:
        ok = test_session(
            args.session_dir,
            verbose=args.verbose,
            processed_root=args.save_processed,
        )
        sys.exit(0 if ok else 1)
    else:
        test_data_dir(
            args.data_dir,
            verbose=args.verbose,
            processed_root=args.save_processed,
        )