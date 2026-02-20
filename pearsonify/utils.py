"""Utilities for Pearsonify: Pearson residuals, confidence intervals, coverage."""

from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import logit
from sklearn.linear_model import LogisticRegression


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


def plot_ccp_curve(
    results: dict,
    model_name: str = "Model",
    n_bins: int = 20,
    output_dir: Optional[str] = None,
    min_x: int = 5,
    min_y: int = 4,
):
    """Plot CCP (calibration) curves. If output_dir is set, save figure there (path to file)."""
    n_methods = len(results["predictions"])
    n_cols = min(3, n_methods)
    n_rows = (n_methods + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(min_x * n_cols, min_y * n_rows), dpi=500
    )
    fig.suptitle(
        model_name,
        y=0.95,
    )  # size/weight from rcParams (figure.titlesize, figure.titleweight)

    if n_methods == 1:
        axes = [axes]
    elif n_rows == 1:
        axes = axes if n_methods > 1 else [axes]
    else:
        axes = axes.flatten()

    plt.subplots_adjust(wspace=0.3, hspace=0.3)
    colors = ["#c430c1", "#ffa94d", "#55d3ed", "#69db7c", "#ff6b6b", "#4ecdc4"]

    for i, (name, preds) in enumerate(results["predictions"].items()):
        preds = np.array(preds)
        y_test = np.array(results["y_test"])
        logits = logit(np.clip(preds, 1e-10, 1 - 1e-10))
        lr = LogisticRegression()
        lr.fit(logits.reshape(-1, 1), y_test)

        pred_grid = np.linspace(preds.min(), preds.max(), n_bins)
        logit_grid = logit(np.clip(pred_grid, 1e-10, 1 - 1e-10))
        observed_rates = lr.predict_proba(logit_grid.reshape(-1, 1))[:, 1]
        pred_means = pred_grid
        error = np.mean(np.abs(observed_rates - pred_means))

        ax = axes[i]
        ax.plot(
            pred_means,
            observed_rates,
            label=f"CE: {error:.2%}",
            color=colors[i % len(colors)],
            marker="o",
            markersize=4,
            linewidth=2,
        )
        ax.set_title(f"{name.replace('_', ' ')}")
        ax.set_xlabel("Predicted probability")
        ax.set_ylabel("Observed event rate")
        ax.grid(True, linestyle="dotted", linewidth=0.7, alpha=0.6)
        ax.plot([0, 1], [0, 1], linestyle="dotted", color="black", alpha=0.5)
        ax.legend(loc="upper left")
        ax.set_xticks(np.arange(0, 1.1, 0.2))
        ax.set_yticks(np.arange(0, 1.1, 0.2))

    for i in range(n_methods, len(axes)):
        axes[i].set_visible(False)

    plt.tight_layout()
    if output_dir:
        plt.savefig(output_dir, bbox_inches="tight", dpi=500)
    plt.close()
    return fig
