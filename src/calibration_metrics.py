"""Reliability diagrams and Expected Calibration Error."""

import numpy as np


def reliability_curve(probs_pos, y, n_bins=10):
    """Bin by predicted probability; return per-bin confidence vs accuracy."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(probs_pos, edges) - 1, 0, n_bins - 1)

    conf, acc, count = [], [], []
    for b in range(n_bins):
        mask = idx == b
        count.append(int(mask.sum()))
        if mask.sum() == 0:
            conf.append(np.nan)
            acc.append(np.nan)
        else:
            conf.append(float(probs_pos[mask].mean()))
            acc.append(float(y[mask].mean()))

    return np.array(conf), np.array(acc), np.array(count)


def ece(probs_pos, y, n_bins=10):
    """Expected Calibration Error: size-weighted mean |confidence - accuracy|."""
    conf, acc, count = reliability_curve(probs_pos, y, n_bins)
    valid = count > 0
    weights = count[valid] / count[valid].sum()
    return float(np.sum(weights * np.abs(conf[valid] - acc[valid])))