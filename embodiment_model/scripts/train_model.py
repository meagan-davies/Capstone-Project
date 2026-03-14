"""
Train embodiment regression model
"""

import argparse
import sys
from pathlib import Path
import yaml

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_loader import load_embodiment_sessions
from src.data.dataset_builder import build_feature_matrix
from src.models.trainer import train_embodiment_model, save_model, compare_models


def main(args):
    """Main training script"""
    
    print("\n" + "="*70)
    print(" "*20 + "EMBODIMENT MODEL TRAINING")
    print("="*70 + "\n")
    
    # Load configuration
    config_path = Path(args.config) if args.config else project_root / "config" / "model_config.yaml"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"✓ Loaded config from {config_path}")
    else:
        config = {}
        print("⚠ No config file found, using defaults")
    
    # Load data
    print(f"\nLoading data from {args.data_dir}...")
    sessions = load_embodiment_sessions(
        args.data_dir,
        participant_ids=args.participants.split(',') if args.participants else None
    )
    
    if len(sessions) == 0:
        print("❌ No sessions loaded. Check data directory.")
        return
    
    # Build feature matrix
    print("\nBuilding feature matrix...")
    X, y, participant_ids, feature_names = build_feature_matrix(sessions)
    
    # Model comparison or single model training
    if args.compare:
        print("\nComparing multiple model types...")
        results = compare_models(X, y, participant_ids, feature_names)
        
        # Save best model
        valid_results = {k: v for k, v in results.items() if v is not None}
        if valid_results:
            best_type = max(valid_results, key=lambda k: valid_results[k]['r2'])
            best_model = valid_results[best_type]['model']
            best_cv = valid_results[best_type]['cv_results']
            
            output_path = Path(args.output) / f"best_model_{best_type}.pkl"
            save_model(best_model, best_cv, output_path, metadata={'comparison': results})
    
    else:
        # Train single model
        model_type = args.model_type or config.get('model', {}).get('type', 'ridge')
        
        print(f"\nTraining {model_type} model...")
        model, cv_results = train_embodiment_model(
            X, y, participant_ids,
            model_type=model_type,
            feature_selection=args.feature_selection,
            n_features=args.n_features or config.get('model', {}).get('feature_selection', {}).get('n_features', 30),
            feature_names=feature_names
        )
        
        # Save model
        output_path = Path(args.output) / f"{args.name}.pkl"
        metadata = {
            'data_dir': str(args.data_dir),
            'n_sessions': len(sessions),
            'n_participants': len(set(participant_ids)),
            'model_type': model_type
        }
        save_model(model, cv_results, output_path, metadata)
    
    print("\n" + "="*70)
    print("Training complete!")
    print("="*70 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train embodiment regression model")
    
    parser.add_argument("--data-dir", type=str, required=True,
                       help="Directory containing training data")
    parser.add_argument("--output", type=str, default="../../artifacts/embodiment_model",
                       help="Output directory for saved model")
    parser.add_argument("--name", type=str, default="embodiment_model",
                       help="Model name")
    parser.add_argument("--model-type", type=str, choices=['ridge', 'lasso', 'random_forest', 'xgboost'],
                       help="Type of regression model")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple model types")
    parser.add_argument("--feature-selection", action="store_true", default=True,
                       help="Perform automatic feature selection")
    parser.add_argument("--n-features", type=int,
                       help="Number of features to select")
    parser.add_argument("--participants", type=str,
                       help="Comma-separated list of participant IDs to include")
    parser.add_argument("--config", type=str,
                       help="Path to config YAML file")
    
    args = parser.parse_args()
    main(args)