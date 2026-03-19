"""
Evaluate trained embodiment model

Usage (from repo root):
    python embodiment_model/scripts/evaluate_model.py \
        --model-path artifacts/embodiment-model/embodiment_model.pkl \
        --test-data  data/raw/embodiment/ \
        --output     artifacts/embodiment-model/eval/ \
        --plot
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# ── path resolution ────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
_src_root    = project_root / "src"
_repo_root   = project_root.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_src_root))   # index 0: wins over any venv `data` package

from data.data_loader    import load_sessions
from data.dataset_builder import build_training_dataset
from models.trainer      import load_model
from models.evaluator    import EmbodimentEvaluator, cross_validate_conditions


def main(args):
    print("\n" + "=" * 70)
    print(" " * 20 + "EMBODIMENT MODEL EVALUATION")
    print("=" * 70 + "\n")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"Loading model from {args.model_path}...")
    model, cv_results, metadata = load_model(args.model_path)

    print("\nModel info:")
    print(f"  Type:                {model.model_type}")
    print(f"  Training CV R²:      {cv_results.get('r2', 'N/A')}")
    print(f"  Training sessions:   {metadata.get('n_sessions', 'N/A')}")
    print(f"  Training participants:{metadata.get('n_participants', 'N/A')}")

    # ── Load test sessions ─────────────────────────────────────────────────
    print(f"\nLoading test data from {args.test_data}...")
    sessions = load_sessions(args.test_data)          # FIXED: was load_embodiment_sessions

    if not sessions:
        print("❌ No test sessions found")
        return

    print(f"✓ {len(sessions)} session(s) loaded")

    # ── Build feature matrix ───────────────────────────────────────────────
    print("\nBuilding feature matrix...")
    X, y = build_training_dataset(sessions)           # FIXED: was build_feature_matrix(sessions)

    participant_ids = np.array(
        [s.participant_id for s in sessions if s.session_id in X.index]
    )
    feature_names = list(X.columns)

    print(f"✓ {X.shape[0]} sessions × {X.shape[1]} features")

    if X.shape[0] < 2:
        print("\n⚠  Only 1 session — evaluation metrics need ≥ 2 samples.")
        print("   Collect more sessions before running evaluation.")
        return

    # ── Evaluate ───────────────────────────────────────────────────────────
    evaluator = EmbodimentEvaluator(model)

    print("\nEvaluating on test set...")
    results = evaluator.evaluate(X.values, y.values, participant_ids)
    evaluator.print_evaluation(results)

    # ── Plots ──────────────────────────────────────────────────────────────
    if args.plot:
        output_dir = Path(args.output) if args.output else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)

        print("\nGenerating plots...")
        save_path = output_dir / "evaluation_results.png" if output_dir else None
        evaluator.plot_results(results, save_path)

        if args.feature_importance:
            save_path = output_dir / "feature_importance.png" if output_dir else None
            evaluator.plot_feature_importance(top_k=20, save_path=save_path)

    # ── Condition breakdown ────────────────────────────────────────────────
    if args.analyze_conditions:
        conditions = np.array([s.condition for s in sessions if s.session_id in X.index])
        print("\n" + "=" * 70)
        print("Performance by Condition")
        print("=" * 70)
        cross_validate_conditions(model, X.values, y.values, conditions)

    print("\n" + "=" * 70)
    print("Evaluation complete!")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained embodiment model")
    parser.add_argument("--model-path",        type=str, required=True,
                        help="Path to trained model .pkl")
    parser.add_argument("--test-data",         type=str, required=True,
                        help="Directory containing test sessions")
    parser.add_argument("--output",            type=str,
                        help="Output directory for plots")
    parser.add_argument("--plot",              action="store_true", default=True,
                        help="Generate evaluation plots")
    parser.add_argument("--feature-importance",action="store_true",
                        help="Plot feature importance")
    parser.add_argument("--analyze-conditions",action="store_true",
                        help="Break down performance by experimental condition")
    args = parser.parse_args()
    main(args)