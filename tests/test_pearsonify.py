import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from pearsonify import Pearsonify

def test_pearsonify():
    # Generate synthetic classification data
    np.random.seed(42)
    X, y = make_classification(n_samples=100, n_features=10, random_state=42)

    # Split into training, calibration, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.4, random_state=42)
    X_cal, X_test, y_cal, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Initialize Pearsonify with an SVM model
    clf = SVC(probability=True, random_state=42)
    model = Pearsonify(estimator=clf, alpha=0.05)

    # Fit the model on training and calibration sets
    model.fit(X_train, y_train, X_cal, y_cal)

    # Generate prediction intervals for test set
    y_test_pred_proba, lower_bounds, upper_bounds = model.predict_intervals(X_test)

    # Validate output shapes
    assert y_test_pred_proba.shape == y_test.shape
    assert lower_bounds.shape == y_test.shape
    assert upper_bounds.shape == y_test.shape

    # Check coverage
    coverage = model.evaluate_coverage(y_test, lower_bounds, upper_bounds)
    assert 0 <= coverage <= 1
