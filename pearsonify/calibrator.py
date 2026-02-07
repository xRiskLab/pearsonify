"""calibrator.py."""

from typing import Protocol

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin


class _ClassifierWithProba(Protocol):
    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class PearsonCalibrator(BaseEstimator, ClassifierMixin):
    """Calibrate probabilities using Pearson residual-informed soft targets."""

    def __init__(
        self,
        base_estimator: _ClassifierWithProba,
        n_bins: int = 10,
        alpha: float = 0.05,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_bins = n_bins
        self.alpha = alpha  # Learning rate for Pearson corrections
        self.bin_targets: dict[int, float] = {}
        self.bin_uncertainties: dict[int, float] = {}

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "PearsonCalibrator":
        """Fit base model and learn Pearson residual-informed calibration."""
        # Train base model
        self.base_estimator.fit(X_train, y_train)

        # Get uncalibrated probabilities on calibration set
        p_uncal = self.base_estimator.predict_proba(X_cal)[:, 1]

        # Create Pearson residual-informed soft targets
        self._create_pearson_targets(y_cal, p_uncal)

        return self

    def _create_pearson_targets(self, y_true: np.ndarray, p_uncal: np.ndarray) -> None:
        """Create calibration targets using Pearson residual information."""
        # Compute Pearson residuals
        residuals = self._compute_pearson_residuals(y_true, p_uncal)

        bin_indices = self._create_bin_indices(p_uncal)
        self.bin_targets = {}

        for i in range(self.n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                n_samples = np.sum(mask)

                # Standard empirical frequency
                empirical_freq = np.mean(y_true[mask])

                # Pearson residual-based correction
                mean_residual = np.mean(residuals[mask])
                std_residual = np.std(residuals[mask])
                mean_prob = np.mean(p_uncal[mask])

                # The key insight: if residuals are positive, the model underestimated
                # So we should increase the calibrated probability
                # Scale by the variance term to make correction proportional to uncertainty
                correction = (
                    self.alpha * mean_residual * np.sqrt(mean_prob * (1 - mean_prob))
                )

                # Teacher target: empirical frequency + Pearson correction
                teacher_target = np.clip(empirical_freq + correction, 0.01, 0.99)

                # Uncertainty estimate for calibration belt
                # Combines binomial uncertainty + residual uncertainty
                binomial_se = np.sqrt(empirical_freq * (1 - empirical_freq) / n_samples)
                residual_se = (
                    std_residual
                    / np.sqrt(n_samples)
                    * np.sqrt(mean_prob * (1 - mean_prob))
                )
                total_se = np.sqrt(binomial_se**2 + (self.alpha * residual_se) ** 2)

                self.bin_targets[i] = teacher_target
                self.bin_uncertainties[i] = total_se
            else:
                self.bin_targets[i] = 0.5
                self.bin_uncertainties[i] = 0.5

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities using learned targets."""
        p_uncal = self.base_estimator.predict_proba(X)[:, 1]

        bin_indices = self._create_bin_indices(p_uncal)
        p_cal = np.array([self.bin_targets[i] for i in bin_indices])
        p_cal = np.clip(p_cal, 1e-10, 1 - 1e-10)

        return np.column_stack([1 - p_cal, p_cal])

    def _create_bin_indices(self, p_uncal: np.ndarray) -> np.ndarray:
        bins = np.linspace(0, 1, self.n_bins + 1)
        result = np.digitize(p_uncal, bins) - 1
        result = np.clip(result, 0, self.n_bins - 1)
        return result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def _compute_pearson_residuals(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> np.ndarray:
        """Compute Pearson residuals."""
        y_pred_proba = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
        return (y_true - y_pred_proba) / np.sqrt(y_pred_proba * (1 - y_pred_proba))
