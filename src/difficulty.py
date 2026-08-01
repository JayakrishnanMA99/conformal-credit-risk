"""Learned difficulty model for normalized conformal scores."""

import numpy as np
from sklearn.model_selection import StratifiedKFold

from src.models import MODELS


def oof_true_label_scores(model_name, X, y, train_idx, seed, n_folds=5):
    """Honest surprise scores on training rows, via K-fold CV.

    Each training row is scored by a model that never saw it, so the
    errors reflect genuine difficulty rather than memorisation.
    """
    X_tr, y_tr = X[train_idx], y[train_idx]
    oof = np.empty(len(train_idx))

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    for fit_i, held_i in skf.split(X_tr, y_tr):
        m = MODELS[model_name](X_tr[fit_i], y_tr[fit_i], seed)
        p = m.predict_proba(X_tr[held_i])
        oof[held_i] = 1.0 - p[np.arange(len(held_i)), y_tr[held_i]]

    return oof

def fit_difficulty(X, oof_scores, train_idx, seed, floor=0.05):
    """Train a regressor predicting the base model's surprise score.

    Deliberately shallow: sigma only needs broad difficulty structure,
    and an over-fitted sigma adds noise without improving sets.
    """
    from lightgbm import LGBMRegressor

    m = LGBMRegressor(
        n_estimators=150,
        learning_rate=0.05,
        num_leaves=15,
        min_child_samples=50,
        random_state=seed,
        verbose=-1,
    )
    m.fit(X[train_idx], oof_scores)

    def sigma(X_new):
        s = m.predict(X_new)
        return np.maximum(s, floor)

    return sigma, m


def make_normalized_score_fn(sigma_vals, base_score_fn):
    """Score function dividing base scores by predicted difficulty.

    sigma_vals : (n,) difficulty for each row, aligned with the probs
                 array that will be passed in.
    """
    def fn(probs):
        return base_score_fn(probs) / sigma_vals[:, None]
    return fn