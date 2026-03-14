"""
Embodiment regression models
"""

import numpy as np
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from typing import Dict, Optional, Tuple


class EmbodimentRegressor:
    """
    Wrapper for embodiment regression models
    """
    # TODO: test the different model types and compare
    def __init__(self, model_type: str = 'ridge', **model_params):
        """
        Initialize regressor
        
        Args:
            model_type: 'ridge', 'lasso', 'elastic_net', 'random_forest', 'xgboost'
            **model_params: Model-specific parameters
        """
        self.model_type = model_type
        self.model = self._create_model(model_type, model_params)
        self.scaler = StandardScaler()
        self.is_fitted = False
        self.feature_names = None
    
    def _create_model(self, model_type: str, params: Dict):
        """Create sklearn/xgboost model"""
        
        if model_type == 'ridge':
            return Ridge(
                alpha=params.get('alpha', 1.0),
                random_state=42
            )
        
        elif model_type == 'lasso':
            return Lasso(
                alpha=params.get('alpha', 0.1),
                max_iter=params.get('max_iter', 10000),
                random_state=42
            )
        
        elif model_type == 'elastic_net':
            return ElasticNet(
                alpha=params.get('alpha', 0.1),
                l1_ratio=params.get('l1_ratio', 0.5),
                max_iter=params.get('max_iter', 10000),
                random_state=42
            )
        
        elif model_type == 'random_forest':
            return RandomForestRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 5),
                min_samples_split=params.get('min_samples_split', 10),
                random_state=42
            )
        
        elif model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=params.get('n_estimators', 100),
                max_depth=params.get('max_depth', 3),
                learning_rate=params.get('learning_rate', 0.05),
                subsample=params.get('subsample', 0.8),
                colsample_bytree=params.get('colsample_bytree', 0.8),
                reg_alpha=params.get('reg_alpha', 0.1),
                reg_lambda=params.get('reg_lambda', 1.0),
                random_state=42
            )
        
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[list] = None):
        """
        Fit the model
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            feature_names: Optional list of feature names
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Fit model
        self.model.fit(X_scaled, y)
        
        self.is_fitted = True
        self.feature_names = feature_names
        
        # Calculate training score
        train_score = self.model.score(X_scaled, y)
        print(f"Training R² = {train_score:.3f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict embodiment scores
        
        Args:
            X: Feature matrix
        
        Returns:
            Predicted embodiment scores (0-100)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        # Clip to valid range
        predictions = np.clip(predictions, 0, 100)
        
        return predictions
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importance (if available)
        
        Returns:
            Array of feature importances or None
        """
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'feature_importances_'):
            # Tree-based models
            return self.model.feature_importances_
        
        elif hasattr(self.model, 'coef_'):
            # Linear models - use absolute coefficients
            return np.abs(self.model.coef_)
        
        else:
            return None
    
    def get_coefficients(self) -> Optional[Dict]:
        """
        Get model coefficients (for linear models)
        
        Returns:
            Dict with intercept and coefficients or None
        """
        if not self.is_fitted:
            return None
        
        if hasattr(self.model, 'coef_') and hasattr(self.model, 'intercept_'):
            return {
                'intercept': float(self.model.intercept_),
                'coefficients': self.model.coef_.tolist(),
                'feature_names': self.feature_names
            }
        
        return None
    
    def extract_formula(self, top_k: int = 10) -> str:
        """
        Extract human-readable formula (for linear models)
        
        Args:
            top_k: Number of top features to include
        
        Returns:
            Formula string
        """
        coeffs = self.get_coefficients()
        
        if coeffs is None:
            return f"Formula extraction not available for {self.model_type}"
        
        intercept = coeffs['intercept']
        coef_values = np.array(coeffs['coefficients'])
        feature_names = coeffs['feature_names'] or [f"x{i}" for i in range(len(coef_values))]
        
        # Sort by absolute coefficient value
        sorted_indices = np.argsort(np.abs(coef_values))[::-1]
        top_indices = sorted_indices[:top_k]
        
        # Build formula
        formula = f"Embodiment = {intercept:.2f}"
        
        for idx in top_indices:
            coef = coef_values[idx]
            name = feature_names[idx]
            sign = '+' if coef > 0 else '-'
            formula += f"\n  {sign} {abs(coef):.3f} * {name}"
        
        return formula


def select_features_lasso(X: np.ndarray, 
                         y: np.ndarray, 
                         feature_names: list,
                         n_features: int = 30) -> Tuple[np.ndarray, list]:
    """
    Select features using Lasso regression
    
    Args:
        X: Feature matrix
        y: Target values
        feature_names: List of feature names
        n_features: Target number of features
    
    Returns:
        Tuple of (selected_X, selected_feature_names)
    """
    from sklearn.linear_model import LassoCV
    
    # TODO: check scaling here makes sense and isn't introducing odd redundancy
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit LassoCV to find optimal alpha
    lasso_cv = LassoCV(cv=5, random_state=42, max_iter=10000)
    lasso_cv.fit(X_scaled, y)
    
    # Get non-zero coefficients
    nonzero_mask = lasso_cv.coef_ != 0
    
    # If too many features, select top ones by coefficient magnitude
    if np.sum(nonzero_mask) > n_features:
        coef_abs = np.abs(lasso_cv.coef_)
        top_indices = np.argsort(coef_abs)[::-1][:n_features]
        mask = np.zeros(len(feature_names), dtype=bool)
        mask[top_indices] = True
    else:
        mask = nonzero_mask
    
    X_selected = X[:, mask]
    selected_names = [name for name, keep in zip(feature_names, mask) if keep]
    
    print(f"Lasso feature selection: {len(selected_names)}/{len(feature_names)} features selected")
    
    return X_selected, selected_names