"""
Evaluate trained embodiment model for pre, post, and prosthetic trials

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
import pandas as pd
import json

# ── path resolution ────────────────────────────────────────────────────────
project_root = Path(__file__).resolve().parent.parent
_src_root    = project_root / "src"
_repo_root   = project_root.parent
sys.path.insert(0, str(_repo_root))
sys.path.insert(0, str(_src_root))

from data.data_loader    import load_sessions
from data.dataset_builder import build_training_dataset
from models.trainer      import load_model
from models.evaluator    import EmbodimentEvaluator

# ── Helper: get ground-truth from labels.json ──────────────────────────────
def load_ground_truth(json_path: Path) -> float:
    with open(json_path, "r") as f:
        data = json.load(f)
    return data.get("embodiment_score", np.nan)

# ── Main Evaluation ───────────────────────────────────────────────────────
def main(args):
    print("\n" + "=" * 70)
    print(" " * 20 + "EMBODIMENT MODEL EVALUATION")
    print("=" * 70 + "\n")

    # Load model
    print(f"Loading model from {args.model_path}...")
    model, cv_results, metadata = load_model(args.model_path)
    evaluator = EmbodimentEvaluator(model)

    print("\nModel info:")
    print(f"  Type:                {model.model_type}")
    print(f"  Training CV R²:      {cv_results.get('r2', 'N/A')}")
    print(f"  Training sessions:   {metadata.get('n_sessions', 'N/A')}")
    print(f"  Training participants:{metadata.get('n_participants', 'N/A')}")

    # Load test sessions
    print(f"\nLoading test data from {args.test_data}...")
    sessions = load_sessions(args.test_data)

    if not sessions:
        print("❌ No test sessions found")
        return
    print(f"✓ {len(sessions)} session(s) loaded")

    # Build feature matrix
    X, y = build_training_dataset(sessions)
    participant_ids = np.array([s.participant_id for s in sessions if s.session_id in X.index])
    print(f"✓ {X.shape[0]} sessions × {X.shape[1]} features")

    if X.shape[0] < 2:
        print("\n⚠  Only 1 session — evaluation metrics need ≥ 2 samples.")
        return

    # Evaluate on test set
    results = evaluator.evaluate(X.values, y.values, participant_ids)
    evaluator.print_evaluation(results)

    # Plot results
    if args.plot:
        output_dir = Path(args.output) if args.output else None
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
        save_path = output_dir / "evaluation_results.png" if output_dir else None
        evaluator.plot_results(results, save_path)
        if args.feature_importance:
            save_path = output_dir / "feature_importance.png" if output_dir else None
            evaluator.plot_feature_importance(top_k=20, save_path=save_path)

    # Compare predicted vs ground truth for prosthetic trials
    print("\n" + "="*70)
    print("Predicted vs Ground Truth for Prosthetic Trials")
    print("="*70)
    comparison_rows = []

    for session in sessions:
        if "pros" in session.session_id.lower():  # Identify prosthetic trials
            X_s, y_s = build_training_dataset([session])
            y_pred = evaluator.model.predict(X_s.values)[0]

            # Use session_dir to locate labels.json
            json_path = session.session_dir / "labels.json"

            y_true = load_ground_truth(json_path)
            comparison_rows.append({
                "participant": session.participant_id,
                "session_id": session.session_id,
                "predicted_score": y_pred,
                "ground_truth_score": y_true,
                "error": y_pred - y_true
            })

    if comparison_rows:
        df_compare = pd.DataFrame(comparison_rows)
        print(df_compare.to_string(index=False))
        if args.output:
            Path(args.output).mkdir(parents=True, exist_ok=True)
            df_compare.to_csv(Path(args.output) / "prosthetic_comparison.csv", index=False)
            print(f"\n✓ Prosthetic comparison saved to {Path(args.output) / 'prosthetic_comparison.csv'}")
    else:
        print("No prosthetic trials found in test data.")
        
    print("\nEvaluation complete!\n" + "="*70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained embodiment model")
    parser.add_argument("--model-path",        type=str, required=True, help="Path to trained model .pkl")
    parser.add_argument("--test-data",         type=str, required=True, help="Directory containing test sessions")
    parser.add_argument("--output",            type=str, help="Output directory for plots and prosthetic comparison")
    parser.add_argument("--plot",              action="store_true", default=True, help="Generate evaluation plots")
    parser.add_argument("--feature-importance", action="store_true", help="Plot feature importance")
    args = parser.parse_args()
    main(args)