import numpy as np
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from pearsonify import PearsonCalibrator


def _get_data():
    np.random.seed(42)
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=5, random_state=42
    )
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.4, random_state=42
    )
    X_cal, X_test, y_cal, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    return X_train, X_cal, X_test, y_train, y_cal, y_test


def test_fit_predict_proba_shapes_and_range():
    """Fit with base_estimator, predict_proba returns (n_samples, 2) and values in [0,1]."""
    X_train, X_cal, X_test, y_train, y_cal, y_test = _get_data()
    base = LogisticRegression(random_state=42)
    cal = PearsonCalibrator(base_estimator=base, n_bins=10)
    cal.fit(X_train, y_train, X_cal, y_cal)

    proba = cal.predict_proba(X_test)
    assert proba.shape == (len(X_test), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)
    assert np.all(proba >= 0) and np.all(proba <= 1)


def test_fit_proba_transform():
    """fit_proba(p_cal, y_cal) then transform(p) without base_estimator."""
    _, X_cal, X_test, _, y_cal, _ = _get_data()
    base = LogisticRegression(random_state=42)
    base.fit(X_cal, y_cal)
    p_cal = base.predict_proba(X_cal)[:, 1]

    cal = PearsonCalibrator(n_bins=10)
    cal.fit_proba(p_cal, y_cal)

    p_test = base.predict_proba(X_test)[:, 1]
    p_calibrated = cal.transform(p_test)
    assert p_calibrated.shape == p_test.shape
    assert np.all(p_calibrated >= 0) and np.all(p_calibrated <= 1)


def test_bin_targets_and_uncertainties_populated():
    """After fit, bin_targets and bin_uncertainties have n_bins entries."""
    X_train, X_cal, _, y_train, y_cal, _ = _get_data()
    n_bins = 8
    cal = PearsonCalibrator(
        base_estimator=LogisticRegression(random_state=42), n_bins=n_bins
    )
    cal.fit(X_train, y_train, X_cal, y_cal)

    assert len(cal.bin_targets) == n_bins
    assert len(cal.bin_uncertainties) == n_bins
    for i in range(n_bins):
        assert 0 <= cal.bin_targets[i] <= 1
        assert cal.bin_uncertainties[i] >= 0


def test_fit_raises_without_base_estimator():
    """fit() requires base_estimator."""
    X_train, X_cal, _, y_train, y_cal, _ = _get_data()
    cal = PearsonCalibrator(base_estimator=None, n_bins=10)
    with pytest.raises(ValueError, match="base_estimator is required"):
        cal.fit(X_train, y_train, X_cal, y_cal)


def test_predict_proba_raises_without_base_estimator():
    """predict_proba(X) requires base_estimator even after fit_proba."""
    _, X_cal, _, _, y_cal, _ = _get_data()
    p_cal = np.random.RandomState(42).uniform(0.2, 0.8, size=len(y_cal))
    cal = PearsonCalibrator(base_estimator=None, n_bins=10)
    cal.fit_proba(p_cal, y_cal)

    _, _, X_test, _, _, _ = _get_data()
    with pytest.raises(ValueError, match="base_estimator is required"):
        cal.predict_proba(X_test)


def test_predict_returns_binary():
    """predict(X) returns 0/1 labels with correct shape."""
    X_train, X_cal, X_test, y_train, y_cal, y_test = _get_data()
    cal = PearsonCalibrator(
        base_estimator=LogisticRegression(random_state=42), n_bins=10
    )
    cal.fit(X_train, y_train, X_cal, y_cal)

    pred = cal.predict(X_test)
    assert pred.shape == (len(X_test),)
    assert np.setdiff1d(np.unique(pred), [0, 1]).size == 0


def test_transform_before_fit_raises():
    """transform(p) before fit or fit_proba raises (bin_targets empty)."""
    cal = PearsonCalibrator(base_estimator=None, n_bins=10)
    p = np.array([0.3, 0.5, 0.7])
    with pytest.raises(KeyError):
        cal.transform(p)


def test_calibrator_with_prefitted_base():
    """fit() works when base_estimator is already fitted."""
    X_train, X_cal, X_test, y_train, y_cal, _ = _get_data()
    base = LogisticRegression(random_state=42)
    base.fit(X_train, y_train)

    cal = PearsonCalibrator(base_estimator=base, n_bins=10)
    cal.fit(X_train, y_train, X_cal, y_cal)

    proba = cal.predict_proba(X_test)
    assert proba.shape[0] == len(X_test)
    assert np.all(proba >= 0) and np.all(proba <= 1)
