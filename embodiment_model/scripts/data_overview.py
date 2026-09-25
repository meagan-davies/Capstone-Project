"""
data_overview.py
---------------
Generate a comprehensive Excel overview of the embodiment dataset.

Sheets
------
Trial Summary
    One row per trial with participant, condition, task, survey scores,
    and predictions from each trained model.

Features
    One row per trial containing the 10 model input features.

Data QC
    Recording sizes, duration, sensor availability, and notes.

Usage
-----
python embodiment_model/scripts/data_overview.py \
    --data-dir data/raw/embodiment \
    --model ridge=artifacts/embodiment_model/ridge/model.pkl \
    --model lasso=artifacts/embodiment_model/lasso/model.pkl \
    --model random_forest=artifacts/embodiment_model/random_forest/model.pkl \
    --model xgboost=artifacts/embodiment_model/xgboost/model.pkl \
    --output analysis/data_summary/embodiment_summary.xlsx
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = PROJECT_ROOT / "embodiment_model" / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import joblib
import pandas as pd

from embodiment_model.src.data.data_loader import load_sessions
from embodiment_model.src.data.dataset_builder import build_feature_matrix

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)


MODEL_DISPLAY_NAMES = {
    "ridge": "Ridge Prediction",
    "lasso": "Lasso Prediction",
    "random_forest": "Random Forest Prediction",
    "xgboost": "XGBoost Prediction",
}

FEATURE_COLUMNS = [
    "eda_mean",
    "eda_std",
    "pinch_mean",
    "pinch_max",
    "grab_mean",
    "palm_speed_mean",
    "palm_speed_max",
    "movement_smoothness",
    "path_length_mm",
    "pinch_events",
]


def parse_model_args(model_args):
    """Load --model name=path arguments."""
    models = {}

    for item in model_args or []:
        if "=" not in item:
            logger.warning(
                "Ignoring malformed model argument: %s", item
            )
            continue

        name, path_str = item.split("=", 1)
        name = name.strip().lower()
        path = Path(path_str.strip())

        if not path.exists():
            logger.warning(
                "Model '%s' not found: %s", name, path
            )
            continue

        models[name] = joblib.load(path)

    return models


def extract_session_features(session):
    """
    Build the same session-level feature vector used by the ML pipeline.
    """

    dataset = build_feature_matrix(session)

    features = (
        dataset.features
        .drop(columns=["elapsed_s"], errors="ignore")
        .mean()
        .reindex(FEATURE_COLUMNS)
    )

    return features


def predict_session(model_artifact, features: pd.Series) -> float:
    """Predict embodiment score for one session using a saved model artifact."""
    X = pd.DataFrame([features])

    if isinstance(model_artifact, dict):
        model = model_artifact["model"]

        # Use the feature order expected by the trained model.
        metadata = model_artifact.get("metadata", {})
        feature_names = metadata.get("feature_names")

        if feature_names is not None:
            X = X.reindex(columns=feature_names)

        # The scaler was fitted without feature names.
        prediction = model.predict(X.to_numpy())[0]

    else:
        prediction = model_artifact.predict(X.to_numpy())[0]

    return float(prediction)


def build_summary(sessions, models):
    """Build Trial Summary, Features, and Data QC tables."""

    trial_rows = []
    feature_rows = []
    qc_rows = []

    for session in sessions:

        session_id = session.session_id

        # ---------------------------------------------------------------
        # Feature extraction
        # ---------------------------------------------------------------
        try:
            features = extract_session_features(session)
        except Exception as exc:
            logger.warning(
                "Feature extraction failed for %s: %s",
                session_id,
                exc,
            )
            features = None

        # ---------------------------------------------------------------
        # Trial Summary
        # ---------------------------------------------------------------
        trial_row = {
            "Session ID": session_id,
            "Participant": session.participant_id,
            "Condition": session.condition_label,
            "Task": session.task_label,
            "Freeze-Check Embodiment Score": session.freeze_check_score,
            "Post-Test Questionnaire Score": session.questionnaire_score,
        }

        # ---------------------------------------------------------------
        # Model predictions
        # ---------------------------------------------------------------
        for model_name, model in models.items():

            column_name = MODEL_DISPLAY_NAMES.get(
                model_name,
                f"{model_name} Prediction",
            )

            if features is None:
                trial_row[column_name] = None
                continue

            try:
                trial_row[column_name] = predict_session(
                    model,
                    features,
                )
            except Exception as exc:
                logger.warning(
                    "Prediction failed for %s using %s: %s",
                    session_id,
                    model_name,
                    exc,
                )
                trial_row[column_name] = None

        trial_rows.append(trial_row)

        # ---------------------------------------------------------------
        # Features
        # ---------------------------------------------------------------
        if features is not None:

            feature_row = {
                "Session ID": session_id,
                "Participant": session.participant_id,
                "Condition": session.condition_label,
                "Task": session.task_label,
                "Freeze-Check Embodiment Score":
                    session.freeze_check_score,
            }

            for feature_name in FEATURE_COLUMNS:
                feature_row[feature_name] = features.get(
                    feature_name
                )

            feature_rows.append(feature_row)

        # ---------------------------------------------------------------
        # Data QC
        # ---------------------------------------------------------------
        qc_row = {
            "Session ID": session_id,
            "Participant": session.participant_id,
            "Condition": session.condition_label,
            "Task": session.task_label,
            "Session Duration (s)": session.duration_s,
            "Leap Motion Samples": len(session.leap_df),
            "BioRadio Samples": len(session.bioradio_df),
            "Apple Watch Available": session.watch_df is not None,
            "Apple Watch Records": (
                len(session.watch_df)
                if session.watch_df is not None
                else 0
            ),
            "Feature Extraction Successful": features is not None,
            "Notes": session.notes,
        }

        qc_rows.append(qc_row)

    return (
        pd.DataFrame(trial_rows),
        pd.DataFrame(feature_rows),
        pd.DataFrame(qc_rows),
    )


def write_excel(
    output_path,
    trial_df,
    feature_df,
    qc_df,
):
    """Write summary tables to a formatted Excel workbook."""

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with pd.ExcelWriter(
        output_path,
        engine="openpyxl",
    ) as writer:

        trial_df.to_excel(
            writer,
            sheet_name="Trial Summary",
            index=False,
        )

        feature_df.to_excel(
            writer,
            sheet_name="Features",
            index=False,
        )

        qc_df.to_excel(
            writer,
            sheet_name="Data QC",
            index=False,
        )

        # ---------------------------------------------------------------
        # Basic formatting
        # ---------------------------------------------------------------
        for sheet_name in [
            "Trial Summary",
            "Features",
            "Data QC",
        ]:

            worksheet = writer.sheets[sheet_name]

            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions

            for column in worksheet.columns:

                max_length = 0
                column_letter = column[0].column_letter

                for cell in column:
                    value = str(cell.value) if cell.value is not None else ""
                    max_length = max(max_length, len(value))

                worksheet.column_dimensions[
                    column_letter
                ].width = min(max_length + 2, 35)


def main():

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--data-dir",
        required=True,
        type=Path,
    )

    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="name=path to trained model .pkl",
    )

    parser.add_argument(
        "--output",
        required=True,
        type=Path,
    )

    args = parser.parse_args()

    # ---------------------------------------------------------------
    # Load models
    # ---------------------------------------------------------------
    models = parse_model_args(args.model)

    print(f"Loading {len(models)} model(s)...")

    # ---------------------------------------------------------------
    # Load sessions using the actual pipeline
    # ---------------------------------------------------------------
    sessions = load_sessions(args.data_dir)

    print(f"Loaded {len(sessions)} session(s).")

    if not sessions:
        raise RuntimeError(
            "No valid embodiment sessions were found."
        )

    # ---------------------------------------------------------------
    # Build tables
    # ---------------------------------------------------------------
    trial_df, feature_df, qc_df = build_summary(
        sessions,
        models,
    )

    # ---------------------------------------------------------------
    # Sort consistently
    # ---------------------------------------------------------------
    sort_columns = [
        "Participant",
        "Condition",
        "Task",
    ]

    trial_df = trial_df.sort_values(
        sort_columns
    )

    feature_df = feature_df.sort_values(
        sort_columns
    )

    qc_df = qc_df.sort_values(
        sort_columns
    )

    # ---------------------------------------------------------------
    # Write workbook
    # ---------------------------------------------------------------
    write_excel(
        args.output,
        trial_df,
        feature_df,
        qc_df,
    )

    print()
    print(f"✓ Wrote summary to: {args.output}")
    print(f"  Trials:   {len(trial_df)}")
    print(f"  Features: {len(feature_df)}")
    print(f"  QC rows:  {len(qc_df)}")


if __name__ == "__main__":
    main()