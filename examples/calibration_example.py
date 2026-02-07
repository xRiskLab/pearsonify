import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.model_selection import train_test_split

from pearsonify import PearsonCalibrator
from pearsonify.utils import plot_ccp_curve

# Generate synthetic data
np.random.seed(42)
X, y = make_classification(
    n_samples=2000, n_features=20, n_informative=10, n_classes=2, random_state=42
)

# Split data: 60% train, 20% cal, 20% test
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42
)
X_cal, X_test, y_cal, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42
)

print(f"Train size: {len(X_train)}, Cal size: {len(X_cal)}, Test size: {len(X_test)}")

# Raw Random Forest (known to be poorly calibrated)
raw_rf = RandomForestClassifier(n_estimators=100, random_state=42)
raw_rf.fit(X_train, y_train)

# Calibrated Random Forest (use same number of bins as plotting)
cal_rf = PearsonCalibrator(
    RandomForestClassifier(n_estimators=100, random_state=42), n_bins=15
)
cal_rf.fit(X_train, y_train, X_cal, y_cal)

# Get predictions
raw_probs = raw_rf.predict_proba(X_test)[:, 1]
cal_probs = cal_rf.predict_proba(X_test)[:, 1]

# Calculate metrics
raw_brier = brier_score_loss(y_test, raw_probs)
cal_brier = brier_score_loss(y_test, cal_probs)
raw_logloss = log_loss(y_test, raw_probs)
cal_logloss = log_loss(y_test, cal_probs)

# Prepare results for plotting
results = {
    "predictions": {"raw_random_forest": raw_probs, "pearson_calibrated": cal_probs},
    "y_test": y_test,
}

# Plot CCP curves
plot_ccp_curve(
    results,
    model_name="Random Forest Calibration Comparison",
    n_bins=15,
    output_dir="ims/Figure_2.png",
)

# Print calibration metrics
print("\nCalibration Metrics:")
print(f"Raw Brier Score: {raw_brier:.4f}")
print(f"Calibrated Brier Score: {cal_brier:.4f}")
print(f"Raw Log Loss: {raw_logloss:.4f}")
print(f"Calibrated Log Loss: {cal_logloss:.4f}")

print("\nPearson Calibrator Details:")
print(f"Bin targets: {[f'{v:.3f}' for v in cal_rf.bin_targets.values()]}")
print(f"Bin uncertainties: {[f'{v:.3f}' for v in cal_rf.bin_uncertainties.values()]}")
