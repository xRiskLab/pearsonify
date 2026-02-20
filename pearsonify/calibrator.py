"""calibrator.py."""

from typing import Protocol, cast

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted


class _ClassifierWithProba(Protocol):
    """Protocol for a classifier with a `fit` and `predict_proba` method."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> object: ...
    def predict_proba(self, X: np.ndarray) -> np.ndarray: ...


class PearsonCalibrator(BaseEstimator, ClassifierMixin):
    """Wrap a classifier and calibrate its probabilities using Pearson residual-informed soft targets."""

    def __init__(
        self,
        base_estimator: _ClassifierWithProba | None = None,
        n_bins: int = 10,
        alpha: float = 0.05,
    ) -> None:
        self.base_estimator = base_estimator
        self.n_bins = n_bins
        self.alpha = alpha
        self.bin_targets: dict[int, float] = {}
        self.bin_uncertainties: dict[int, float] = {}

    def fit_proba(self, p_cal: np.ndarray, y_cal: np.ndarray) -> "PearsonCalibrator":
        """Learn calibration from raw model's probabilities on the calibration set.
        Use when you already have raw probabilities and don't need a base_estimator.
        """
        self._create_pearson_targets(y_cal, p_cal)
        return self

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "PearsonCalibrator":
        """Fit base_estimator (if not already fitted), then learn Pearson calibration
        from its output on X_cal. The calibrator wraps and corrects the base_estimator's
        own probabilities, it does not fit a separate model.
        """
        if self.base_estimator is None:
            raise ValueError(
                "base_estimator is required for fit(); use fit_proba(p_cal, y_cal) to calibrate raw probabilities directly."
            )
        try:
            check_is_fitted(cast(BaseEstimator, self.base_estimator))
        except NotFittedError:
            self.base_estimator.fit(X_train, y_train)

        # Use base_estimator's own output as the uncalibrated probabilities to correct
        p_uncal = self.base_estimator.predict_proba(X_cal)[:, 1]
        self._create_pearson_targets(y_cal, p_uncal)
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        """Apply learned Pearson calibration to raw probabilities."""
        bin_indices = self._create_bin_indices(p)
        p_cal = np.array([self.bin_targets[i] for i in bin_indices])
        return np.clip(p_cal, 1e-10, 1 - 1e-10)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return calibrated probabilities.
        Gets base_estimator's raw output, then applies Pearson correction.
        """
        if self.base_estimator is None:
            raise ValueError(
                "base_estimator is required for predict_proba(X); use transform(p) to calibrate raw probabilities directly."
            )
        # Get raw probs from the wrapped model, then correct them
        p_uncal = self.base_estimator.predict_proba(X)[:, 1]
        p_cal = self.transform(p_uncal)
        return np.column_stack([1 - p_cal, p_cal])

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary predictions."""
        return (self.predict_proba(X)[:, 1] > 0.5).astype(int)

    def _create_pearson_targets(self, y_true: np.ndarray, p_uncal: np.ndarray) -> None:
        """Create calibration targets using Pearson residual information."""
        residuals = self._compute_pearson_residuals(y_true, p_uncal)
        bin_indices = self._create_bin_indices(p_uncal)
        self.bin_targets = {}

        for i in range(self.n_bins):
            mask = bin_indices == i
            if np.sum(mask) > 0:
                n_samples = np.sum(mask)
                empirical_freq = np.mean(y_true[mask])
                mean_residual = np.mean(residuals[mask])
                std_residual = np.std(residuals[mask])
                mean_prob = np.mean(p_uncal[mask])

                correction = (
                    self.alpha * mean_residual * np.sqrt(mean_prob * (1 - mean_prob))
                )
                teacher_target = np.clip(empirical_freq + correction, 0.01, 0.99)

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

    def _create_bin_indices(self, p_uncal: np.ndarray) -> np.ndarray:
        """Create bin indices for the given probabilities."""
        bins = np.linspace(0, 1, self.n_bins + 1)
        result = np.digitize(p_uncal, bins) - 1
        return np.clip(result, 0, self.n_bins - 1)

    def _compute_pearson_residuals(
        self, y_true: np.ndarray, y_pred_proba: np.ndarray
    ) -> np.ndarray:
        """Compute Pearson residuals."""
        y_pred_proba = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
        return (y_true - y_pred_proba) / np.sqrt(y_pred_proba * (1 - y_pred_proba))
