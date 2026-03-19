"""
Train embodiment regression model
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root.parent))  # repo root for shared/

from src.data.data_loader    import load_sessions               # was: load_embodiment_sessions
from src.data.dataset_builder import build_training_dataset      # was: build_feature_matrix (wrong signature)
from src.models.trainer      import train_embodiment_model, save_model, compare_models


def main(args):
    print("\n" + "="*70)
    print(" "*20 + "EMBODIMENT MODEL TRAINING")
    print("="*70 + "\n")

    # Load configuration
    config_path = Path(args.config) if args.config else project_root / "config" / "model_config.yaml"
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

    sessions = load_sessions(                                   # FIXED: was load_embodiment_sessions
        data_dir,
        participant_ids=args.participants.split(",") if args.participants else None,
    )

    if not sessions:
        print("❌ No sessions loaded. Check data directory and labels.json files.")
        return

    print(f"✓ Loaded {len(sessions)} session(s)")

    # ── Build feature matrix ───────────────────────────────────────────────
    print("\nBuilding feature matrix...")

    # FIXED: was build_feature_matrix(sessions) — wrong function, wrong signature.
    # build_feature_matrix() takes ONE EmbodimentSession → AlignedDataset.
    # build_training_dataset() takes a LIST → (X DataFrame, y Series).
    X, y = build_training_dataset(sessions)                     # FIXED

    participant_ids = [s.participant_id for s in sessions if s.session_id in X.index]
    feature_names   = list(X.columns)

    print(f"✓ Feature matrix: {X.shape[0]} sessions × {X.shape[1]} features")
    print(f"  Embodiment score range: [{y.min():.1f}, {y.max():.1f}]")

    # ── Train ──────────────────────────────────────────────────────────────
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.compare:
        print("\nComparing multiple model types...")
        results = compare_models(
            X.values, y.values, participant_ids, feature_names,
        )

        valid = {k: v for k, v in results.items() if v is not None}
        if valid:
            best_type = max(valid, key=lambda k: valid[k]["r2"])
            best_model  = valid[best_type]["model"]
            best_cv     = valid[best_type]["cv_results"]
            output_path = output_dir / f"best_model_{best_type}.pkl"
            save_model(best_model, best_cv, output_path,
                       metadata={"comparison": {k: v for k, v in results.items() if v}})

    else:
        model_type = args.model_type or config.get("model", {}).get("type", "ridge")
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

        output_path = output_dir / f"{args.name}.pkl"
        save_model(model, cv_results, output_path, metadata={
            "data_dir":       str(data_dir),
            "n_sessions":     len(sessions),
            "n_participants": len(set(participant_ids)),
            "model_type":     model_type,
            "features":       feature_names,
        })

    print("\n" + "="*70)
    print("Training complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train embodiment regression model")
    parser.add_argument("--data-dir",         type=str, required=True)
    parser.add_argument("--output",           type=str,
                        default="../../artifacts/embodiment-model")
    parser.add_argument("--name",             type=str, default="embodiment_model")
    parser.add_argument("--model-type",       type=str,
                        choices=["ridge", "lasso", "random_forest", "xgboost"])
    parser.add_argument("--compare",          action="store_true")
    parser.add_argument("--feature-selection",action="store_true", default=True)
    parser.add_argument("--n-features",       type=int)
    parser.add_argument("--participants",     type=str)
    parser.add_argument("--config",           type=str)
    args = parser.parse_args()
    main(args)