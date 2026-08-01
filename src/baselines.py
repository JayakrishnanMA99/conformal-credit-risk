"""Calibration baselines: raw thresholding, Platt, isotonic."""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression


def sets_from_probs(probs, alpha):
    """Keep every label with probability >= alpha.

    The naive reading of '90% confidence': anything with at least
    10% probability stays in.
    """
    return probs >= alpha


def platt_calibrate(cal_probs, cal_y, test_probs):
    """Fit Platt scaling on calibration data, apply to test."""
    lr = LogisticRegression(C=1e10)
    lr.fit(cal_probs[:, [1]], cal_y)
    p1 = lr.predict_proba(test_probs[:, [1]])[:, 1]
    return np.column_stack([1 - p1, p1])


def isotonic_calibrate(cal_probs, cal_y, test_probs):
    """Fit isotonic regression on calibration data, apply to test."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(cal_probs[:, 1], cal_y)
    p1 = iso.predict(test_probs[:, 1])
    return np.column_stack([1 - p1, p1])