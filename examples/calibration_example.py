"""calibration_example.py."""

from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

from pearsonify import PearsonCalibrator
from pearsonify.utils import plot_ccp_curve

# Store _ for unused imports
_, _, _ = (
    LogisticRegression,
    RandomForestClassifier,
    MLPClassifier,
)

plt.rcParams.update(
    {
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Latin Modern Math"],
        "font.weight": "bold",
        "axes.titlesize": 14,  # subplot titles (each panel)
        "axes.titleweight": "bold",
        "figure.titlesize": 18,  # figure suptitle (top title only)
        "figure.titleweight": "bold",
        "axes.labelsize": 14,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 14,
        "figure.dpi": 800,
        "axes.linewidth": 0.8,
        "font.size": 14,
    }
)

# Raw model to wrap and calibrate.
# MODEL = (RandomForestClassifier, {"n_estimators": 100, "random_state": 42})
# MODEL = (LogisticRegression, {"random_state": 42})
MODEL = (MLPClassifier, {"hidden_layer_sizes": (100,), "random_state": 42})
N_BINS = 15

# Generate synthetic data
np.random.seed(42)
X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=10, n_classes=2, random_state=42
)

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)
print(f"Train size: {len(X_train)}, Cal size: {len(X_cal)}, Test size: {len(X_test)}")

# Fit raw model
raw_clf, raw_kwargs = MODEL
raw_classifier = raw_clf(**raw_kwargs)
raw_classifier.fit(X_train, y_train)
raw_name = type(raw_classifier).__name__
raw_probs = raw_classifier.predict_proba(X_test)[:, 1]

# Wrap raw model in calibrator and learn Pearson correction from its cal set output
calibrator = PearsonCalibrator(base_estimator=raw_classifier, n_bins=N_BINS)
calibrator.fit(X_train, y_train, X_cal, y_cal)
cal_probs = calibrator.predict_proba(X_test)[:, 1]
uncalibrated_name = f"{raw_name} (Uncalibrated)"
calibrated_name = f"{raw_name} (Pearson Calibrated)"

# Metrics
raw_brier = brier_score_loss(y_test, raw_probs)
cal_brier = brier_score_loss(y_test, cal_probs)
raw_logloss = log_loss(y_test, raw_probs)
cal_logloss = log_loss(y_test, cal_probs)

results = {
    "predictions": {uncalibrated_name: raw_probs, calibrated_name: cal_probs},
    "y_test": y_test,
}

# Save next to this script's directory: repo/.tmp/paper/ (works from examples/ or repo root)
_script_dir = Path(__file__).resolve().parent
output_dir: str | None = str(
    _script_dir.parent / ".tmp" / "paper" / f"ccp_{raw_name}.png"
)
plot_ccp_curve(
    results,
    model_name=f"CCP Curves for {raw_name}",
    n_bins=N_BINS,
    output_dir=output_dir,
)

print("\nCalibration Metrics:")
print(f"Raw Brier Score: {raw_brier:.4f}")
print(f"Calibrated Brier Score: {cal_brier:.4f}")
print(f"Raw Log Loss: {raw_logloss:.4f}")
print(f"Calibrated Log Loss: {cal_logloss:.4f}")

print("\nPearson Calibrator Details:")
print(f"Bin targets: {[f'{v:.3f}' for v in calibrator.bin_targets.values()]}")
print(
    f"Bin uncertainties:{[f'{v:.3f}' for v in calibrator.bin_uncertainties.values()]}"
)
