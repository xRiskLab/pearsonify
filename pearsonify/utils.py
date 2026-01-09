import numpy as np


def compute_pearson_residuals(y_true, y_pred_proba):
    """Compute Pearson residuals for binary classification."""
    y_pred_proba = np.clip(y_pred_proba, 1e-10, 1 - 1e-10)
    return (y_true - y_pred_proba) / np.sqrt(y_pred_proba * (1 - y_pred_proba))


def compute_confidence_intervals(y_pred_proba, q_alpha):
    """Compute confidence intervals based on Pearson residuals."""
    std_error = np.sqrt(y_pred_proba * (1 - y_pred_proba))
    lower_bounds = np.maximum(0, y_pred_proba - q_alpha * std_error)
    upper_bounds = np.minimum(1, y_pred_proba + q_alpha * std_error)
    return lower_bounds, upper_bounds


def calculate_coverage(y_true, lower_bounds, upper_bounds):
    """Calculate the empirical coverage of confidence intervals."""
    return np.mean((y_true >= lower_bounds) & (y_true <= upper_bounds))
