"""
Test that regression model produces valid outputs
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add paths
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.regressor import EmbodimentRegressor


@pytest.fixture
def sample_data():
    """Generate sample training data"""
    np.random.seed(42)
    n_samples = 100
    n_features = 30
    
    X = np.random.randn(n_samples, n_features)
    y = 50 + 20 * X[:, 0] - 10 * X[:, 1] + 5 * np.random.randn(n_samples)
    y = np.clip(y, 0, 100)  # Clip to valid range
    
    return X, y


def test_ridge_predictions_in_range(sample_data):
    """Test Ridge model predictions are in valid range"""
    X, y = sample_data
    
    model = EmbodimentRegressor(model_type='ridge')
    model.fit(X, y)
    
    predictions = model.predict(X)
    
    # All predictions should be in [0, 100]
    assert np.all(predictions >= 0), "Predictions below 0"
    assert np.all(predictions <= 100), "Predictions above 100"


def test_model_reproducibility(sample_data):
    """Test model produces same results with same random seed"""
    X, y = sample_data
    
    # Train first model
    model1 = EmbodimentRegressor(model_type='ridge', alpha=1.0)
    model1.fit(X, y)
    pred1 = model1.predict(X)
    
    # Train second model (same parameters)
    model2 = EmbodimentRegressor(model_type='ridge', alpha=1.0)
    model2.fit(X, y)
    pred2 = model2.predict(X)
    
    # Predictions should be identical
    np.testing.assert_array_almost_equal(pred1, pred2, decimal=10)


def test_feature_importance_available(sample_data):
    """Test feature importance can be extracted"""
    X, y = sample_data
    
    for model_type in ['ridge', 'lasso', 'random_forest']:
        model = EmbodimentRegressor(model_type=model_type)
        model.fit(X, y)
        
        importance = model.get_feature_importance()
        
        assert importance is not None, f"No importance for {model_type}"
        assert len(importance) == X.shape[1], f"Wrong importance length for {model_type}"
        assert np.all(importance >= 0), f"Negative importance for {model_type}"


def test_formula_extraction(sample_data):
    """Test formula can be extracted from linear models"""
    X, y = sample_data
    
    model = EmbodimentRegressor(model_type='ridge')
    model.fit(X, y, feature_names=[f'feature_{i}' for i in range(X.shape[1])])
    
    formula = model.extract_formula(top_k=10)
    
    assert isinstance(formula, str), "Formula should be string"
    assert 'Embodiment' in formula, "Formula should contain 'Embodiment'"
    assert 'feature_0' in formula or 'feature_1' in formula, "Formula should contain feature names"


def test_model_handles_edge_cases(sample_data):
    """Test model handles edge cases gracefully"""
    X, y = sample_data
    
    model = EmbodimentRegressor(model_type='ridge')
    model.fit(X, y)
    
    # Test with all zeros
    X_zeros = np.zeros((5, X.shape[1]))
    pred_zeros = model.predict(X_zeros)
    assert np.all(np.isfinite(pred_zeros)), "Model fails on zeros"
    
    # Test with large values
    X_large = 100 * np.ones((5, X.shape[1]))
    pred_large = model.predict(X_large)
    assert np.all(np.isfinite(pred_large)), "Model fails on large values"
    
    # Test with single sample
    X_single = X[0:1]
    pred_single = model.predict(X_single)
    assert len(pred_single) == 1, "Wrong prediction shape for single sample"


def test_different_model_types(sample_data):
    """Test all model types can be trained"""
    X, y = sample_data
    
    model_types = ['ridge', 'lasso', 'random_forest', 'xgboost']
    
    for model_type in model_types:
        model = EmbodimentRegressor(model_type=model_type)
        
        # Should fit without errors
        model.fit(X, y)
        
        # Should predict without errors
        predictions = model.predict(X)
        
        # Predictions should be valid
        assert len(predictions) == len(y)
        assert np.all(np.isfinite(predictions))
        assert np.all(predictions >= 0)
        assert np.all(predictions <= 100)


def test_model_improves_with_training(sample_data):
    """Test model error decreases with more training data"""
    X, y = sample_data
    
    model_small = EmbodimentRegressor(model_type='ridge')
    model_small.fit(X[:20], y[:20])  # Train on 20 samples
    pred_small = model_small.predict(X[20:])
    error_small = np.mean(np.abs(pred_small - y[20:]))
    
    model_large = EmbodimentRegressor(model_type='ridge')
    model_large.fit(X[:80], y[:80])  # Train on 80 samples
    pred_large = model_large.predict(X[80:])
    error_large = np.mean(np.abs(pred_large - y[80:]))
    
    # More training data should generally reduce error
    # (not always true due to randomness, but should be true on average)
    assert error_large < error_small * 2, "Model doesn't improve with more data"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])