import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator
from .utils import (
    compute_pearson_residuals,
    compute_confidence_intervals,
    calculate_coverage,
)


class Pearsonify:
    def __init__(self, estimator: BaseEstimator, alpha=0.05):
        """
        Initialize with a model that implements `fit` and `predict_proba`.

        Parameters:
        - estimator: A scikit-learn-like classifier with `fit` and `predict_proba` methods.
        - alpha: Significance level (e.g., 0.05 for 95% intervals).
        """
        self.estimator = estimator
        self.alpha = alpha
        self.q_alpha = None

    def fit(self, X_train, y_train, X_cal, y_cal):
        """Fit the model and compute Pearson residual-based quantile from calibration data."""
        # Train the model if it's not already fitted
        self.estimator.fit(X_train, y_train)

        # Compute residuals on calibration set
        y_cal_pred_proba = self.estimator.predict_proba(X_cal)[:, 1]
        residuals = compute_pearson_residuals(y_cal, y_cal_pred_proba)
        self.q_alpha = np.quantile(np.abs(residuals), 1 - self.alpha)
        return self.q_alpha

    def predict_intervals(self, X_test):
        """Generate prediction intervals for new data."""
        if self.q_alpha is None:
            raise ValueError(
                "The model needs to be fitted before predicting intervals."
            )

        # Generate predicted probabilities for the test set
        y_test_pred_proba = self.estimator.predict_proba(X_test)[:, 1]
        lower_bounds, upper_bounds = compute_confidence_intervals(
            y_test_pred_proba, self.q_alpha
        )
        return y_test_pred_proba, lower_bounds, upper_bounds

    def evaluate_coverage(self, y_test, lower_bounds, upper_bounds):
        """Evaluate the empirical coverage."""
        return calculate_coverage(y_test, lower_bounds, upper_bounds)

    def plot_intervals(self, y_test_pred_proba, lower_bounds, upper_bounds, y_test=None):
        """Plot the predicted probabilities with their confidence intervals."""
        if y_test is not None:
            coverage = self.evaluate_coverage(y_test, lower_bounds, upper_bounds)
        sorted_indices = np.argsort(y_test_pred_proba)
        plt.figure(figsize=(10, 6))
        plt.plot(
            y_test_pred_proba[sorted_indices],
            color="dodgerblue",
            label="Predicted Probability"
            + (f" (Coverage: {coverage:.0%})" if y_test is not None else ""),
        )
        plt.fill_between(
            range(len(y_test_pred_proba)),
            lower_bounds[sorted_indices],
            upper_bounds[sorted_indices],
            color="lightblue",
            alpha=0.4,
        )
        plt.title(
            f"Confidence Intervals with Pearsonify\n"
            f"Confidence Level: {(1 - self.alpha):.0%}"
        )
        plt.xlabel("Sorted Test Sample Index")
        plt.ylabel("Probability")
        plt.legend()
        plt.show()
