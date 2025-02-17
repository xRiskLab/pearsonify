import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from pearsonify import Pearsonify

# Generate synthetic classification data
np.random.seed(42)
X, y = make_classification(
    n_samples=1000, n_features=20, n_informative=10, n_classes=2, random_state=42
)

# Split data into train, calibration, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Initialize Pearsonify with an SVC model
clf = SVC(probability=True, random_state=42)
model = Pearsonify(estimator=clf, alpha=0.05)

# Fit the model (trains on training set and calculates q_alpha from calibration set)
model.fit(X_train, y_train, X_cal, y_cal)

# Generate prediction intervals for the test set
y_test_pred_proba, lower_bounds, upper_bounds = model.predict_intervals(X_test)

# Calculate coverage
coverage = model.evaluate_coverage(y_test, lower_bounds, upper_bounds)
print(f"Coverage: {coverage:.2%}")

# Plot the intervals
model.plot_intervals(y_test_pred_proba, lower_bounds, upper_bounds, y_test)