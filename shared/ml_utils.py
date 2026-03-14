"""
ML utilities for both projects
"""
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def leave_one_subject_out_cv(model, X, y, subject_ids, verbose=True):
    """
    Leave-One-Subject-Out cross-validation
    
    Args:
        model: sklearn-compatible model
        X: Feature matrix
        y: Target values
        subject_ids: Array of subject IDs for each sample
        verbose: Print progress
    
    Returns:
        Dict with scores and predictions
    """
    logo = LeaveOneGroupOut()
    
    y_pred_all = np.zeros_like(y)
    scores = []
    
    for i, (train_idx, test_idx) in enumerate(logo.split(X, y, subject_ids)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        y_pred_all[test_idx] = y_pred
        
        score = r2_score(y_test, y_pred)
        scores.append(score)
        
        if verbose:
            subject = subject_ids[test_idx[0]]
            print(f"Fold {i+1}: Subject {subject}, R² = {score:.3f}")
    
    overall_r2 = r2_score(y, y_pred_all)
    overall_mae = mean_absolute_error(y, y_pred_all)
    overall_rmse = np.sqrt(mean_squared_error(y, y_pred_all))
    
    if verbose:
        print("\nOverall LOSO-CV:")
        print(f"  R² = {overall_r2:.3f} (+/- {np.std(scores):.3f})")
        print(f"  MAE = {overall_mae:.2f}")
        print(f"  RMSE = {overall_rmse:.2f}")
    
    return {
        'r2': overall_r2,
        'mae': overall_mae,
        'rmse': overall_rmse,
        'fold_scores': scores,
        'predictions': y_pred_all
    }

# TODO: Incorporate robust scaling for control model,
#       also test the efficacy of using the different ones.
def scale_features(X_train, X_test=None, method='standard'):
    """
    Scale features
    
    Args:
        X_train: Training features
        X_test: Optional test features
        method: 'standard' or 'minmax'
    
    Returns:
        Scaled features and fitted scaler
    """
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    
    if X_test is not None:
        X_test_scaled = scaler.transform(X_test)
        return X_train_scaled, X_test_scaled, scaler
    
    return X_train_scaled, scaler


def print_feature_importance(feature_names, importances, top_k=10):
    """
    Print feature importance
    
    Args:
        feature_names: List of feature names
        importances: Array of importance values
        top_k: Number of top features to print
    """
    sorted_idx = np.argsort(importances)[::-1]
    
    print(f"\nTop {top_k} Features:")
    for i in range(min(top_k, len(feature_names))):
        idx = sorted_idx[i]
        print(f"  {i+1}. {feature_names[idx]}: {importances[idx]:.3f}")