"""
data_summary.py
----------------
Generate a summary CSV of embodiment survey scores and per-model predictions
for every trial in the embodiment dataset.

Output columns:
    Participant Number | Arm Type | Task | Freeze-Check Embodiment Score | Post-Test Questionnaire Score |
    Embodiment Model (Ridge) | Embodiment Model (Random Forest) | Embodiment Model (XGBoost)
    (one prediction column is added per --model you pass in)

Usage:
    python -m embodiment_model.scripts.data_overview 
        --data-dir data/raw/embodiment   
        --model ridge=artifacts/embodiment_model/ridge/embodiment_model.pkl  
        --model lasso=artifacts/embodiment_model/lasso/embodiment_model.pkl  
        --model random_forest=artifacts/embodiment_model/random_forest/embodiment_model.pkl  
        --model xgboost=artifacts/embodiment_model/xgboost/embodiment_model.pkl  
        --output analysis/data_summary/embodiment_summary.xlsx

You can pass zero, one, or several --model flags. With zero, you get just
the survey-score table.
"""
import argparse
import json
import re
import sys
from pathlib import Path

import joblib
import pandas as pd


try:
    from embodiment_model.src.data.data_loader import EmbodimentSession
    from embodiment_model.src.data.parsers import (
        apple_watch_parser,
        bioradio_parser,
        leapmotion_parser,
    )

except ImportError:
    load_trial_sensors = None
    build_feature_vector = None


ARM_TYPE_MAP = {
    "pre": "Biological Before",
    "pros": "Prosthetic",
    "post": "Biological After",
}

TASK_MAP = {
    1: "1 - Grasp",
    2: "2 - Zip",
    3: "3 - Block",
}

CONDITION_ORDER = {"pre": 0, "pros": 1, "post": 2}

# Matches e.g. T01_pre_trial001, P01_post_trial002, T01_pros_trial003
TRIAL_DIR_PATTERN = re.compile(
    r"^(?P<participant>[A-Za-z]*\d+)_(?P<condition>pre|pros|post)_trial(?P<trial_num>\d+)$",
    re.IGNORECASE,
)

MODEL_DISPLAY_NAMES = {
    "ridge": "Embodiment Model (Ridge)",
    "random_forest": "Embodiment Model (Random Forest)",
    "xgboost": "Embodiment Model (XGBoost)",
}


def parse_trial_dir_name(dirname: str):
    m = TRIAL_DIR_PATTERN.match(dirname)
    if not m:
        return None
    return m.group("participant"), m.group("condition").lower()


def load_labels(trial_dir: Path) -> dict:
    labels_path = trial_dir / "labels.json"
    if not labels_path.exists():
        raise FileNotFoundError(f"No labels.json in {trial_dir}")
    with open(labels_path) as f:
        return json.load(f)


def build_features_for_trial(trial_dir: Path):
    """
    Build the feature vector for a single trial. Replace the body of this
    function if your real pipeline doesn't match the placeholder imports.
    """
    if load_trial_sensors is None or build_feature_vector is None:
        raise RuntimeError(
            "Feature extraction pipeline not found. Update the import "
            "block at the top of this script to point at your real "
            "data loader / feature builder modules."
        )
    sensors = load_trial_sensors(trial_dir)
    return build_feature_vector(sensors)


def parse_model_args(model_args):
    """Parse --model name=path into {name: loaded_model}."""
    models = {}
    for item in model_args or []:
        if "=" not in item:
            print(f"Warning: ignoring malformed --model value '{item}' "
                  f"(expected name=path)", file=sys.stderr)
            continue
        name, path_str = item.split("=", 1)
        name = name.strip().lower()
        path = Path(path_str.strip())
        if not path.exists():
            print(f"Warning: model path for '{name}' not found ({path}); "
                  f"its column will be left blank.", file=sys.stderr)
            continue
        models[name] = joblib.load(path)
    return models


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument(
        "--model", action="append", default=[],
        help="name=path to a trained model .pkl, e.g. ridge=artifacts/ridge/model.pkl. "
             "Repeat for multiple models.",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    models = parse_model_args(args.model)

    trial_dirs = sorted(p for p in args.data_dir.iterdir() if p.is_dir())
    rows = []
    skipped = []
    feature_cache = {}  # trial_dir -> features, computed once even with multiple models

    for trial_dir in trial_dirs:
        parsed = parse_trial_dir_name(trial_dir.name)
        if parsed is None:
            skipped.append(trial_dir.name)
            continue
        participant_folder, condition = parsed

        try:
            labels = load_labels(trial_dir)
        except FileNotFoundError as e:
            print(f"Warning: {e}", file=sys.stderr)
            continue

        participant_id = labels.get("participant_id", participant_folder)
        trial_number = labels.get("trial_number")
        freeze_check_score = labels.get("embodiment_score")
        questionnaire_score = labels.get("questionnaire")

        arm_type = ARM_TYPE_MAP.get(condition, condition)
        task_label = TASK_MAP.get(trial_number, f"{trial_number} - Unknown")

        row = {
            "Participant Number": participant_id,
            "Arm Type": arm_type,
            "Task": task_label,
            "Freeze-Check Embodiment Score": freeze_check_score,
            "Post-Test Questionnaire Score": questionnaire_score,
            "_condition_order": CONDITION_ORDER.get(condition, 3),
            "_task_order": trial_number if trial_number is not None else 99,
        }

        if models:
            try:
                if trial_dir not in feature_cache:
                    feature_cache[trial_dir] = build_features_for_trial(trial_dir)
                features = feature_cache[trial_dir]
            except Exception as e:
                print(f"Warning: feature extraction failed for {trial_dir.name}: {e}",
                      file=sys.stderr)
                features = None

            for model_name, model in models.items():
                col_name = MODEL_DISPLAY_NAMES.get(
                    model_name, f"Embodiment Model ({model_name})"
                )
                if features is None:
                    row[col_name] = None
                    continue
                try:
                    row[col_name] = float(model.predict([features])[0])
                except Exception as e:
                    print(f"Warning: prediction failed for {trial_dir.name} "
                          f"with model '{model_name}': {e}", file=sys.stderr)
                    row[col_name] = None

        rows.append(row)

    if skipped:
        print(f"Skipped {len(skipped)} folders that didn't match the naming "
              f"pattern, e.g. {skipped[:3]}", file=sys.stderr)

    if not rows:
        print("No trials found — check --data-dir.", file=sys.stderr)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df.sort_values(
        ["Participant Number", "_condition_order", "_task_order"]
    ).drop(columns=["_condition_order", "_task_order"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"Wrote {len(df)} rows to {args.output}")


if __name__ == "__main__":
    main()