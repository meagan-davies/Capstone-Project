"""
ML utilities for both projects
"""
import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def leave_one_subject_out_cv(model, X, y, subject_ids, feature_names=None, feature_selection=False, n_features=30, verbose=True):
    """
    Leave-One-Subject-Out cross-validation.

    Scaling and optional feature selection are performed separately
    within each training fold to prevent information leakage from the
    held-out subject.

    Args:
        model: sklearn-compatible model.
        X: Feature matrix.
        y: Target values.
        subject_ids: Array of subject IDs for each sample.
        feature_names: Names corresponding to columns of X.
        feature_selection: Whether to perform Lasso feature selection
            within each training fold.
        n_features: Maximum number of features to retain.
        verbose: Print progress.

    Returns:
        Dict with scores, predictions, and selected features.
    """
    logo = LeaveOneGroupOut()

    y_pred_all = np.zeros_like(y)
    scores = []
    selected_features_by_fold = {}

    # Import here to avoid creating a circular import at module level.
    if feature_selection:
        from embodiment_model.src.models.regressor import get_lasso_feature_mask

    for i, (train_idx, test_idx) in enumerate(
        logo.split(X, y, subject_ids)
    ):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # ---------------------------------------------------------
        # 1. Scale using training data only
        # ---------------------------------------------------------
        scaler = StandardScaler()

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # ---------------------------------------------------------
        # 2. Feature selection using training data only
        # ---------------------------------------------------------
        if feature_selection:
            feature_mask = get_lasso_feature_mask(
                X_train_scaled,
                y_train,
                n_features=n_features
            )

            X_train_model = X_train_scaled[:, feature_mask]
            X_test_model = X_test_scaled[:, feature_mask]

            if feature_names is not None:
                selected_names = [
                    name
                    for name, keep in zip(feature_names, feature_mask)
                    if keep
                ]
            else:
                selected_names = None

            selected_features_by_fold[i + 1] = selected_names

        else:
            X_train_model = X_train_scaled
            X_test_model = X_test_scaled

        # ---------------------------------------------------------
        # 3. Create a fresh model for this fold
        # ---------------------------------------------------------
        fold_model = clone(model)

        fold_model.fit(X_train_model, y_train)

        # ---------------------------------------------------------
        # 4. Predict the held-out subject
        # ---------------------------------------------------------
        y_pred = fold_model.predict(X_test_model)

        # Keep predictions within the valid embodiment score range.
        y_pred = np.clip(y_pred, 0, 100)

        y_pred_all[test_idx] = y_pred

        # ---------------------------------------------------------
        # 5. Calculate fold performance
        # ---------------------------------------------------------
        score = r2_score(y_test, y_pred)
        scores.append(score)

        if verbose:
            subject = subject_ids[test_idx[0]]

            if feature_selection:
                n_selected = np.sum(feature_mask)
                print(
                    f"Fold {i + 1}: Subject {subject}, "
                    f"R² = {score:.3f}, "
                    f"Features = {n_selected}"
                )
            else:
                print(
                    f"Fold {i + 1}: Subject {subject}, "
                    f"R² = {score:.3f}"
                )

    # -------------------------------------------------------------
    # Overall out-of-fold performance
    # -------------------------------------------------------------
    overall_r2 = r2_score(y, y_pred_all)
    overall_mae = mean_absolute_error(y, y_pred_all)
    overall_rmse = np.sqrt(
        mean_squared_error(y, y_pred_all)
    )

    if verbose:
        print("\nOverall LOSO-CV:")
        print(
            f"  R² = {overall_r2:.3f} "
            f"(+/- {np.std(scores):.3f})"
        )
        print(f"  MAE = {overall_mae:.2f}")
        print(f"  RMSE = {overall_rmse:.2f}")

    return {
        'r2': overall_r2,
        'mae': overall_mae,
        'rmse': overall_rmse,
        'fold_scores': scores,
        'predictions': y_pred_all,
        'selected_features_by_fold': selected_features_by_fold
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