"""
Train embodiment regression model

Usage (from repo root):
    python embodiment_model/scripts/train_model.py \
        --data-dir   data/raw/embodiment/ \
        --output     artifacts/embodiment-model/ \
        --model-type ridge

    # Compare all model types and save the best:
    python embodiment_model/scripts/train_model.py \
        --data-dir data/raw/embodiment/ \
        --output   artifacts/embodiment-model/ \
        --compare
"""

import argparse
import sys
from pathlib import Path
import yaml

# ── path resolution ────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent   # embodiment_model/
_src_root    = project_root / "src"
_repo_root   = project_root.parent                      # Capstone-Project/
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_src_root))   # index 0: wins over any venv `data` package

from data.data_loader    import load_sessions
from data.dataset_builder import build_training_dataset
from models.trainer      import train_embodiment_model, save_model, compare_models


def main(args):
    print("\n" + "=" * 70)
    print(" " * 20 + "EMBODIMENT MODEL TRAINING")
    print("=" * 70 + "\n")

    # ── Config ────────────────────────────────────────────────────────────
    config_path = (
        Path(args.config) if args.config
        else project_root / "config" / "model_config.yaml"
    )
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded config from {config_path}")
    else:
        config = {}
        print("⚠ No config file found, using defaults")

    # ── Load sessions ──────────────────────────────────────────────────────
    data_dir = Path(args.data_dir)
    print(f"\nLoading sessions from {data_dir}...")

    sessions = load_sessions(
        data_dir,
        participant_ids=args.participants.split(",") if args.participants else None,
    )

    if not sessions:
        print("❌ No sessions loaded. Check data directory and labels.json files.")
        return

    print(f"✓ {len(sessions)} session(s) loaded")

    # ── Feature matrix ────────────────────────────────────────────────────
    print("\nBuilding feature matrix...")
    X, y = build_training_dataset(sessions)

    participant_ids = [s.participant_id for s in sessions if s.session_id in X.index]
    feature_names   = list(X.columns)

    print(f"✓ {X.shape[0]} sessions × {X.shape[1]} features")
    print(f"  Score range: [{y.min():.1f}, {y.max():.1f}]")

    if X.shape[0] < 2:
        print("\n⚠  Only 1 session — cannot run cross-validation.")
        print("   Collect more sessions before training, or use --no-cv flag.")
        print("   Pipeline is working correctly — nothing else to fix.")
        return

    # ── Train ──────────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        print("\nComparing all model types...")
        results = compare_models(X.values, y.values, participant_ids, feature_names)

        valid = {k: v for k, v in results.items() if v is not None}
        if valid:
            best_type  = max(valid, key=lambda k: valid[k]["r2"])
            best_model = valid[best_type]["model"]
            best_cv    = valid[best_type]["cv_results"]
            out_path   = output_dir / f"best_model_{best_type}.pkl"
            save_model(best_model, best_cv, out_path,
                       metadata={"comparison": {k: v for k, v in results.items() if v}})

    else:
        model_type = (
            args.model_type
            or config.get("model", {}).get("type", "ridge")
        )
        n_features = (
            args.n_features
            or config.get("model", {}).get("feature_selection", {}).get("n_features", 30)
        )

        print(f"\nTraining {model_type} model...")
        model, cv_results = train_embodiment_model(
            X.values, y.values, participant_ids,
            model_type=model_type,
            feature_selection=args.feature_selection,
            n_features=n_features,
            feature_names=feature_names,
        )

        out_path = output_dir / f"{args.name}.pkl"
        save_model(model, cv_results, out_path, metadata={
            "data_dir":       str(data_dir),
            "n_sessions":     len(sessions),
            "n_participants": len(set(participant_ids)),
            "model_type":     model_type,
            "features":       feature_names,
        })

    print("\n" + "=" * 70)
    print("Training complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train embodiment regression model")
    parser.add_argument("--data-dir",          type=str, required=True,
                        help="Root data directory  (e.g. data/raw/embodiment/)")
    parser.add_argument("--output",            type=str,
                        default="artifacts/embodiment-model",
                        help="Output directory for saved model")
    parser.add_argument("--name",              type=str, default="embodiment_model",
                        help="Model filename (without .pkl)")
    parser.add_argument("--model-type",        type=str,
                        choices=["ridge", "lasso", "random_forest", "xgboost"],
                        help="Regression model type (default: ridge)")
    parser.add_argument("--compare",           action="store_true",
                        help="Train all model types and save the best")
    parser.add_argument("--feature-selection", action="store_true", default=True,
                        help="Run Lasso feature selection before training")
    parser.add_argument("--n-features",        type=int,
                        help="Max features to keep after selection (default: 30)")
    parser.add_argument("--participants",       type=str,
                        help="Comma-separated participant IDs to include")
    parser.add_argument("--config",            type=str,
                        help="Path to model_config.yaml")
    args = parser.parse_args()
    main(args)