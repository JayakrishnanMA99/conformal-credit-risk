"""Distribution shift experiments: subgroup splits and weighted conformal."""

import numpy as np

AGE_COL = 4  # X5 is AGE (0-indexed)


def make_probabilistic_shift_split(X, y, seed, col=AGE_COL, cut=40,
                                   p_high=0.75, p_low=0.25, train_frac=0.6):
    """Soft covariate shift: both groups span the full feature range.

    Clients above `cut` go to test with probability p_high, below it with
    p_low. Support overlaps, so the density ratio is finite everywhere and
    importance weighting is well-posed.
    """
    rng = np.random.default_rng(seed)

    p_test = np.where(X[:, col] >= cut, p_high, p_low)
    to_test = rng.random(len(y)) < p_test

    test_idx = np.where(to_test)[0]
    rest = rng.permutation(np.where(~to_test)[0])
    n_train = int(train_frac * len(rest))

    return {
        "train": rest[:n_train],
        "cal": rest[n_train:],
        "test": test_idx,
    }
def fit_domain_classifier(X, cal_idx, test_idx, seed):
    """Learn to tell calibration rows from test rows.

    Returns w(x) = c(x) / (1 - c(x)), the density ratio p_test / p_cal
    up to a constant that cancels during normalisation.
    """
    from lightgbm import LGBMClassifier

    X_dom = np.vstack([X[cal_idx], X[test_idx]])
    y_dom = np.concatenate([np.zeros(len(cal_idx)), np.ones(len(test_idx))])

    m = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
        random_state=seed,
        verbose=-1,
    )
    m.fit(X_dom, y_dom)

    def weights(X_new, clip=20.0):
        c = np.clip(m.predict_proba(X_new)[:, 1], 1e-6, 1 - 1e-6)
        return np.clip(c / (1.0 - c), 1.0 / clip, clip)

    return weights, m


def weighted_conformal_quantile(cal_scores, cal_weights, alpha):
    """Weighted (1-alpha) quantile, with the test point's own mass included.

    The test point contributes weight 1 in normalised terms, which is the
    weighted analogue of the (n+1) correction.
    """
    order = np.argsort(cal_scores)
    s = cal_scores[order]
    w = cal_weights[order]

    total = w.sum() + 1.0          # +1 for the test point itself
    cumw = np.cumsum(w) / total

    idx = np.searchsorted(cumw, 1.0 - alpha, side="left")
    if idx >= len(s):
        return np.inf
    return s[idx]