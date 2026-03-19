"""
Model training utilities
"""

import numpy as np
import sys
from pathlib import Path
from typing import Dict, Tuple, Optional
import pickle

# Add shared to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "shared"))
from ml_utils import leave_one_subject_out_cv

from .regressor import EmbodimentRegressor, select_features_lasso

# TODO: change for preferred model type
def train_embodiment_model(X: np.ndarray,
                          y: np.ndarray,
                          participant_ids: np.ndarray,
                          model_type: str = 'ridge',
                          feature_selection: bool = True,
                          n_features: int = 30,
                          feature_names: Optional[list] = None,
                          **model_params) -> Tuple[EmbodimentRegressor, Dict]:
    """
    Train embodiment regression model with cross-validation
    
    Args:
        X: Feature matrix
        y: Embodiment scores
        participant_ids: Participant IDs for LOSO-CV
        model_type: Type of model to train
        feature_selection: Whether to perform feature selection
        n_features: Number of features to select
        feature_names: Optional feature names
        **model_params: Additional model parameters
    
    Returns:
        Tuple of (trained_model, cv_results)
    """
    print(f"\n{'='*60}")
    print(f"Training {model_type.upper()} Embodiment Model")
    print(f"{'='*60}")
    print(f"Data shape: {X.shape}")
    print(f"Participants: {len(np.unique(participant_ids))}")
    print(f"Embodiment range: [{y.min():.1f}, {y.max():.1f}]")
    
    # Feature selection
    X_train = X
    selected_feature_names = feature_names
    
    if feature_selection and X.shape[1] > n_features:
        print(f"\nPerforming feature selection (target: {n_features} features)...")
        X_train, selected_feature_names = select_features_lasso(
            X, y, feature_names, n_features
        )
        print(f"Selected features: {len(selected_feature_names)}")
    
    # Create model
    model = EmbodimentRegressor(model_type=model_type, **model_params)
    
    # Cross-validation
    print(f"\nRunning Leave-One-Subject-Out Cross-Validation...")
    cv_results = leave_one_subject_out_cv(model.model, X_train, y, participant_ids)
    
    # Train final model on all data
    print(f"\nTraining final model on all data...")
    model.fit(X_train, y, feature_names=selected_feature_names)
    
    # Feature importance
    importance = model.get_feature_importance()
    if importance is not None and selected_feature_names is not None:
        print(f"\nTop 10 Most Important Features:")
        sorted_idx = np.argsort(importance)[::-1]
        for i in range(min(10, len(sorted_idx))):
            idx = sorted_idx[i]
            print(f"  {i+1}. {selected_feature_names[idx]}: {importance[idx]:.3f}")
    
    # Extract formula (if linear model)
    if model_type in ['ridge', 'lasso', 'elastic_net']:
        print(f"\n{'-'*60}")
        print("Interpretable Formula:")
        print(f"{'-'*60}")
        print(model.extract_formula(top_k=10))
        print(f"{'-'*60}")
    
    print(f"\n{'='*60}")
    print(f"Training Complete")
    print(f"  Final R² (LOSO-CV): {cv_results['r2']:.3f}")
    print(f"  MAE: {cv_results['mae']:.2f}")
    print(f"  RMSE: {cv_results['rmse']:.2f}")
    print(f"{'='*60}\n")
    
    return model, cv_results


def save_model(model: EmbodimentRegressor,
              cv_results: Dict,
              filepath: Path,
              metadata: Optional[Dict] = None):
    """
    Save trained model to disk
    
    Args:
        model: Trained EmbodimentRegressor
        cv_results: Cross-validation results
        filepath: Output file path
        metadata: Optional metadata dict
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    save_dict = {
        'model': model,
        'cv_results': cv_results,
        'metadata': metadata or {}
    }
    
    with open(filepath, 'wb') as f:
        pickle.dump(save_dict, f)
    
    print(f"✓ Model saved to {filepath}")


def load_model(filepath: Path) -> Tuple[EmbodimentRegressor, Dict, Dict]:
    """
    Load trained model from disk
    
    Args:
        filepath: Model file path
    
    Returns:
        Tuple of (model, cv_results, metadata)
    """
    with open(filepath, 'rb') as f:
        save_dict = pickle.load(f)
    
    model = save_dict['model']
    cv_results = save_dict.get('cv_results', {})
    metadata = save_dict.get('metadata', {})
    
    print(f"✓ Model loaded from {filepath}")
    print(f"  Model type: {model.model_type}")
    print(f"  CV R²: {cv_results.get('r2', 'N/A')}")
    
    return model, cv_results, metadata


def compare_models(X: np.ndarray,
                  y: np.ndarray,
                  participant_ids: np.ndarray,
                  feature_names: Optional[list] = None) -> Dict:
    """
    Compare different model types
    
    Args:
        X: Feature matrix
        y: Target values
        participant_ids: Participant IDs
        feature_names: Feature names
    
    Returns:
        Dict of model results
    """
    model_types = ['ridge', 'lasso', 'random_forest', 'xgboost']
    results = {}
    
    print(f"\n{'='*60}")
    print("Comparing Model Types")
    print(f"{'='*60}\n")
    
    for model_type in model_types:
        print(f"\nTraining {model_type}...")
        
        try:
            model, cv_results = train_embodiment_model(
                X, y, participant_ids,
                model_type=model_type,
                feature_selection=True,
                n_features=30,
                feature_names=feature_names
            )
            
            results[model_type] = {
                'model': model,
                'cv_results': cv_results,
                'r2': cv_results['r2'],
                'mae': cv_results['mae'],
                'rmse': cv_results['rmse']
            }
            
        except Exception as e:
            print(f"⚠ Error training {model_type}: {e}")
            results[model_type] = None
    
    # Print comparison
    print(f"\n{'='*60}")
    print("Model Comparison Results")
    print(f"{'='*60}")
    print(f"{'Model':<20} {'R²':<10} {'MAE':<10} {'RMSE':<10}")
    print(f"{'-'*60}")
    
    for model_type, result in results.items():
        if result is not None:
            print(f"{model_type:<20} {result['r2']:<10.3f} {result['mae']:<10.2f} {result['rmse']:<10.2f}")
    
    print(f"{'='*60}\n")
    
    # Find best model
    valid_results = {k: v for k, v in results.items() if v is not None}
    if valid_results:
        best_model_type = max(valid_results, key=lambda k: valid_results[k]['r2'])
        print(f"✓ Best model: {best_model_type} (R² = {valid_results[best_model_type]['r2']:.3f})")
    
    return results