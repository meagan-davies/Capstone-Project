"""
Evaluate trained embodiment model
"""

import argparse
import sys
from pathlib import Path
import numpy as np

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import load_embodiment_sessions
from src.data.dataset_builder import build_feature_matrix
from src.models.trainer import load_model
from src.models.evaluator import EmbodimentEvaluator, cross_validate_conditions


def main(args):
    """Main evaluation script"""
    
    print("\n" + "="*70)
    print(" "*20 + "EMBODIMENT MODEL EVALUATION")
    print("="*70 + "\n")
    
    # Load model
    print(f"Loading model from {args.model_path}...")
    model, cv_results, metadata = load_model(args.model_path)
    
    print("\nModel Info:")
    print(f"  Type: {model.model_type}")
    print(f"  Training CV R²: {cv_results.get('r2', 'N/A')}")
    if metadata:
        print(f"  Training sessions: {metadata.get('n_sessions', 'N/A')}")
        print(f"  Training participants: {metadata.get('n_participants', 'N/A')}")
    
    # Load test data
    print(f"\nLoading test data from {args.test_data}...")
    sessions = load_embodiment_sessions(args.test_data)
    
    if len(sessions) == 0:
        print("❌ No test sessions found")
        return
    
    # Build feature matrix
    print("\nBuilding feature matrix...")
    X, y, participant_ids, feature_names = build_feature_matrix(sessions)
    
    # Create evaluator
    evaluator = EmbodimentEvaluator(model)
    
    # Evaluate
    print("\nEvaluating model on test set...")
    results = evaluator.evaluate(X, y, participant_ids)
    
    # Print results
    evaluator.print_evaluation(results)
    
    # Plot results
    if args.plot:
        print("\nGenerating plots...")
        save_path = Path(args.output) / "evaluation_results.png" if args.output else None
        evaluator.plot_results(results, save_path)
        
        # Feature importance
        if args.feature_importance:
            save_path = Path(args.output) / "feature_importance.png" if args.output else None
            evaluator.plot_feature_importance(top_k=20, save_path=save_path)
    
    # Condition analysis
    if args.analyze_conditions:
        conditions = np.array([s.condition for s in sessions])
        print("\n" + "="*70)
        print("Performance by Condition")
        print("="*70)
        condition_results = cross_validate_conditions(model, X, y, conditions)
    
    # Control accuracy analysis
    if args.analyze_control:
        # Find control accuracy feature
        control_idx = None
        for i, name in enumerate(model.feature_names or []):
            if 'tracking_error' in name or 'control' in name:
                control_idx = i
                break
        
        if control_idx is not None:
            print("\n" + "="*70)
            print("Control Accuracy Relationship")
            print("="*70)
            evaluator.analyze_control_accuracy_relationship(X, y, control_idx)
    
    print("\n" + "="*70)
    print("Evaluation complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate embodiment model")
    
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model (.pkl)")
    parser.add_argument("--test-data", type=str, required=True,
                       help="Directory containing test data")
    parser.add_argument("--output", type=str,
                       help="Output directory for plots and results")
    parser.add_argument("--plot", action="store_true", default=True,
                       help="Generate evaluation plots")
    parser.add_argument("--feature-importance", action="store_true",
                       help="Plot feature importance")
    parser.add_argument("--analyze-conditions", action="store_true",
                       help="Analyze performance by experimental condition")
    parser.add_argument("--analyze-control", action="store_true",
                       help="Analyze relationship with control accuracy")
    
    args = parser.parse_args()
    main(args)