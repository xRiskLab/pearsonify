import numpy as np
from pearsonify.utils import (
    compute_pearson_residuals,
    compute_confidence_intervals,
    calculate_coverage
)

def test_compute_pearson_residuals():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0.2, 0.7, 0.6, 0.1])
    residuals = compute_pearson_residuals(y_true, y_pred)
    assert residuals.shape == y_true.shape
    assert np.isfinite(residuals).all()

def test_confidence_intervals():
    y_pred = np.array([0.2, 0.7, 0.6, 0.1])
    q_alpha = 1.96
    lb, ub = compute_confidence_intervals(y_pred, q_alpha)
    assert lb.shape == y_pred.shape and ub.shape == y_pred.shape
    assert np.all(lb >= 0) and np.all(ub <= 1)

def test_calculate_coverage():
    y_true = np.array([0, 1, 1, 0])
    lb = np.array([0, 0.6, 0.5, 0])
    ub = np.array([0.3, 1, 0.7, 0.2])
    coverage = calculate_coverage(y_true, lb, ub)
    assert 0 <= coverage <= 1
    assert isinstance(coverage, float)
