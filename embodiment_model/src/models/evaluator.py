"""
Model evaluation and validation utilities
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from typing import Dict, Optional
from pathlib import Path
from scipy.stats import ttest_rel

from .regressor import EmbodimentRegressor


class EmbodimentEvaluator:
    """
    Evaluate embodiment model performance
    """

    def __init__(self, model: EmbodimentRegressor):
        self.model = model

    # =========================================================
    # CORE EVALUATION
    # =========================================================

    def evaluate(self,
                 X: np.ndarray,
                 y_true: np.ndarray,
                 participant_ids: Optional[np.ndarray] = None) -> Dict:

        y_pred = self.model.predict(X)

        results = {
            'r2': r2_score(y_true, y_pred),
            'mae': mean_absolute_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'correlation': np.corrcoef(y_true, y_pred)[0, 1],
            'mean_error': np.mean(y_pred - y_true),
            'std_error': np.std(y_pred - y_true),
            'y_true': y_true,
            'y_pred': y_pred
        }

        if participant_ids is not None:
            results['per_participant'] = self._evaluate_per_participant(
                y_true, y_pred, participant_ids
            )

        return results

    def _evaluate_per_participant(self,
                                 y_true: np.ndarray,
                                 y_pred: np.ndarray,
                                 participant_ids: np.ndarray) -> Dict:

        per_participant = {}

        for pid in np.unique(participant_ids):
            mask = participant_ids == pid

            if np.sum(mask) >= 2:
                per_participant[pid] = {
                    'r2': r2_score(y_true[mask], y_pred[mask]),
                    'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
                    'n_samples': int(np.sum(mask))
                }

        return per_participant

    # =========================================================
    # CONDITION-BASED ANALYSIS
    # =========================================================

    def evaluate_by_condition(self,
                              X: np.ndarray,
                              y_true: np.ndarray,
                              conditions: np.ndarray) -> Dict:

        y_pred = self.model.predict(X)
        results = {}

        for condition in np.unique(conditions):
            mask = conditions == condition

            if np.sum(mask) < 2:
                continue

            results[condition] = {
                'r2': r2_score(y_true[mask], y_pred[mask]),
                'mae': mean_absolute_error(y_true[mask], y_pred[mask]),
                'mean_true': float(np.mean(y_true[mask])),
                'mean_pred': float(np.mean(y_pred[mask])),
                'n_samples': int(np.sum(mask))
            }

        return results

    # =========================================================
    # EMBODIMENT DELTAS (KEY FEATURE)
    # =========================================================

    def compute_embodiment_deltas(self,
                                  y_pred: np.ndarray,
                                  conditions: np.ndarray,
                                  participant_ids: np.ndarray) -> Dict:

        results = {}

        for pid in np.unique(participant_ids):
            mask = participant_ids == pid

            conds = conditions[mask]
            preds = y_pred[mask]

            pdata = {}

            for c in ['pre', 'post', 'prosthetic']:
                if c in conds:
                    pdata[c] = float(np.mean(preds[conds == c]))

            if all(k in pdata for k in ['pre', 'post']):
                pdata['delta_post_pre'] = pdata['post'] - pdata['pre']

            if all(k in pdata for k in ['pre', 'prosthetic']):
                pdata['delta_prosthetic_pre'] = pdata['prosthetic'] - pdata['pre']

            if all(k in pdata for k in ['post', 'prosthetic']):
                pdata['delta_prosthetic_post'] = pdata['prosthetic'] - pdata['post']

            results[pid] = pdata

        return results

    # =========================================================
    # FINAL PROSTHETIC SCORE
    # =========================================================

    def compute_final_prosthetic_score(self,
                                       y_pred: np.ndarray,
                                       conditions: np.ndarray,
                                       participant_ids: np.ndarray) -> Dict:

        scores = {}

        for pid in np.unique(participant_ids):
            mask = (participant_ids == pid) & (conditions == 'prosthetic')

            if np.sum(mask) > 0:
                scores[pid] = float(np.mean(y_pred[mask]))

        return scores

    # =========================================================
    # STATISTICAL COMPARISON
    # =========================================================

    def compare_conditions_statistically(self,
                                         y_pred: np.ndarray,
                                         conditions: np.ndarray,
                                         participant_ids: np.ndarray):

        pre_vals = []
        prosthetic_vals = []

        for pid in np.unique(participant_ids):
            mask = participant_ids == pid

            conds = conditions[mask]
            preds = y_pred[mask]

            if 'pre' in conds and 'prosthetic' in conds:
                pre_vals.append(np.mean(preds[conds == 'pre']))
                prosthetic_vals.append(np.mean(preds[conds == 'prosthetic']))

        if len(pre_vals) >= 2:
            stat, p = ttest_rel(pre_vals, prosthetic_vals)
            print(f"Pre vs Prosthetic: t={stat:.3f}, p={p:.5f}")

    # =========================================================
    # PRINTING
    # =========================================================

    def print_evaluation(self, results: Dict):

        print(f"\n{'='*60}")
        print("Model Evaluation Results")
        print(f"{'='*60}")
        print(f"R² Score:       {results['r2']:.3f}")
        print(f"MAE:            {results['mae']:.2f}")
        print(f"RMSE:           {results['rmse']:.2f}")
        print(f"Correlation:    {results['correlation']:.3f}")
        print(f"Mean Error:     {results['mean_error']:.2f}")
        print(f"Std Error:      {results['std_error']:.2f}")

        if 'per_participant' in results:
            print(f"\nPer-Participant Performance:")
            print(f"{'Participant':<15} {'R²':<10} {'MAE':<10} {'N':<5}")
            print(f"{'-'*45}")

            for pid, metrics in results['per_participant'].items():
                print(f"{pid:<15} {metrics['r2']:<10.3f} {metrics['mae']:<10.2f} {metrics['n_samples']:<5}")

        print(f"{'='*60}\n")

    # =========================================================
    # PLOTTING
    # =========================================================

    def plot_results(self, results: Dict, save_path: Optional[Path] = None):

        y_true = results['y_true']
        y_pred = results['y_pred']

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Predicted vs actual
        axes[0, 0].scatter(y_true, y_pred, alpha=0.6)
        axes[0, 0].plot([0, 100], [0, 100], '--')
        axes[0, 0].set_title("Predicted vs Actual")

        # Residuals
        residuals = y_pred - y_true
        axes[0, 1].scatter(y_pred, residuals, alpha=0.6)
        axes[0, 1].axhline(0, linestyle='--')

        # Histogram
        axes[1, 0].hist(residuals, bins=20)
        axes[1, 0].axvline(0, linestyle='--')

        # Empty / placeholder
        axes[1, 1].axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300)

        plt.show()

    def plot_condition_comparison(self,
                                 y_pred: np.ndarray,
                                 conditions: np.ndarray):

        unique_conditions = np.unique(conditions)
        data = [y_pred[conditions == c] for c in unique_conditions]

        plt.figure(figsize=(8, 6))
        plt.boxplot(data, labels=unique_conditions)

        plt.ylabel("Predicted Embodiment Score")
        plt.title("Embodiment Across Conditions")
        plt.grid(alpha=0.3)

        plt.show()

    def plot_feature_importance(self,
                                top_k: int = 20,
                                save_path: Optional[Path] = None):

        importance = self.model.get_feature_importance()

        if importance is None:
            print("Feature importance not available")
            return

        names = self.model.feature_names or [f"F{i}" for i in range(len(importance))]
        idx = np.argsort(importance)[::-1][:top_k]

        plt.figure(figsize=(10, 6))
        plt.barh(range(len(idx)), importance[idx])
        plt.yticks(range(len(idx)), [names[i] for i in idx])
        plt.gca().invert_yaxis()

        if save_path:
            plt.savefig(save_path, dpi=300)

        plt.show()

def cross_validate_conditions(model: EmbodimentRegressor,
                              X: np.ndarray,
                              y: np.ndarray,
                              conditions: np.ndarray) -> dict:
    """
    Evaluate model performance across experimental conditions
    """
    results = {}
    for condition in np.unique(conditions):
        mask = conditions == condition
        if np.sum(mask) < 2:
            continue
        X_cond = X[mask]
        y_cond = y[mask]
        y_pred = model.predict(X_cond)
        results[condition] = {
            "r2": r2_score(y[mask], y_pred),
            "mae": mean_absolute_error(y[mask], y_pred),
            "n_samples": np.sum(mask),
            "mean_true": np.mean(y[mask]),
            "mean_pred": np.mean(y_pred)
        }
    return results