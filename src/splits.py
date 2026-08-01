"""Three-way splitting: train / calibration / test."""

import numpy as np
from sklearn.model_selection import train_test_split


def make_split(y, seed, cal_frac=0.2, test_frac=0.2, stratify=True):
    """Split row indices into train/calibration/test.

    Returns a dict with keys 'train', 'cal', 'test' (arrays of indices).
    """
    n = len(y)
    idx = np.arange(n)
    strat = y if stratify else None

    # First cut: hold out (cal + test) together
    holdout_frac = cal_frac + test_frac
    train_idx, rest_idx = train_test_split(
        idx, test_size=holdout_frac, random_state=seed, stratify=strat
    )

    # Second cut: divide the holdout into cal and test
    rest_strat = y[rest_idx] if stratify else None
    cal_idx, test_idx = train_test_split(
        rest_idx,
        test_size=test_frac / holdout_frac,
        random_state=seed,
        stratify=rest_strat,
    )

    splits = {"train": train_idx, "cal": cal_idx, "test": test_idx}

    # No row may appear in two splits
    all_idx = np.concatenate(list(splits.values()))
    assert len(np.unique(all_idx)) == n, "splits overlap or lose rows"

    return splits