"""Split conformal prediction, from scratch."""

import numpy as np


def scores_lac(probs):
    """LAC / THR score: 1 - p(label), for every label.

    probs : (n, K) predicted probabilities
    returns : (n, K) scores. Low score = model finds this label plausible.
    """
    return 1.0 - probs


def true_label_scores(probs, y):
    """Pick out the score of each row's actual label.

    Used on calibration data, where the truth is known.
    """
    all_scores = scores_lac(probs)
    return all_scores[np.arange(len(y)), y]

def conformal_quantile(cal_scores, alpha):
    """The (1-alpha) conformal quantile with finite-sample correction.

    cal_scores : (n,) surprise scores on calibration data (true labels)
    alpha : miss rate budget, e.g. 0.1 for 90% coverage
    returns : scalar threshold q_hat
    """
    n = len(cal_scores)
    k = int(np.ceil((n + 1) * (1 - alpha)))

    if k > n:
        return np.inf   # too few calibration points for this alpha

    return np.sort(cal_scores)[k - 1]

def build_sets(probs, q_hat, score_fn=scores_lac):
    """Prediction sets: keep every label with score <= q_hat.

    returns : (n, K) boolean array
    """
    return score_fn(probs) <= q_hat


def evaluate(sets, y):
    """Coverage and set-size summary."""
    covered = sets[np.arange(len(y)), y]
    sizes = sets.sum(axis=1)

    return {
        "coverage": covered.mean(),
        "avg_size": sizes.mean(),
        "frac_empty": (sizes == 0).mean(),
        "frac_singleton": (sizes == 1).mean(),
        "frac_full": (sizes == 2).mean(),
    }

def evaluate_by_class(sets, y):
    """Coverage and set size, broken down by the client's true class."""
    out = {}
    for k in np.unique(y):
        mask = y == k
        out[int(k)] = {
            "n": int(mask.sum()),
            "coverage": float(sets[mask, k].mean()),
            "avg_size": float(sets[mask].sum(axis=1).mean()),
        }
    return out

def mondrian_quantiles(cal_probs, cal_y, alpha, score_fn=scores_lac):
    """One conformal quantile per class, computed from that class only."""
    all_scores = score_fn(cal_probs)
    q = {}
    for k in np.unique(cal_y):
        mask = cal_y == k
        q[int(k)] = conformal_quantile(all_scores[mask, k], alpha)
    return q


def build_sets_mondrian(probs, q_by_class, score_fn=scores_lac):
    """Keep label k iff its score is <= that class's own threshold."""
    all_scores = score_fn(probs)
    sets = np.zeros_like(all_scores, dtype=bool)
    for k, qk in q_by_class.items():
        sets[:, k] = all_scores[:, k] <= qk
    return sets

def scores_aps(probs, rng=None):
    """APS: cumulative probability mass swept to reach each label.

    Sort labels descending; a label's score is the total mass of all
    labels at least as likely as it, including itself.

    Randomization (subtracting a uniform fraction of the label's own
    mass) is REQUIRED here, not optional. With two classes the second
    label's cumulative mass is always exactly 1.0, so deterministic APS
    produces a large block of tied scores at 1.0 and q_hat collapses to
    1.0 — every set becomes {both}. Randomization breaks the ties.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)

    u = rng.random(size=(probs.shape[0], 1))
    cumsum = cumsum - u * sorted_p

    scores = np.empty_like(probs)
    np.put_along_axis(scores, order, cumsum, axis=1)
    return scores


def scores_raps(probs, k_reg=1, lam=0.1, rng=None):
    """RAPS: randomized APS plus a penalty on labels with poor rank.

    The penalty discourages the long tail of low-probability labels that
    APS tends to include. With only two classes the penalty applies to at
    most one label per row, so RAPS and APS behave very similarly here;
    the difference becomes meaningful on multi-class problems.
    """
    if rng is None:
        rng = np.random.default_rng(0)

    order = np.argsort(-probs, axis=1)
    sorted_p = np.take_along_axis(probs, order, axis=1)
    cumsum = np.cumsum(sorted_p, axis=1)

    u = rng.random(size=(probs.shape[0], 1))
    cumsum = cumsum - u * sorted_p

    ranks = np.arange(1, probs.shape[1] + 1)
    cumsum = cumsum + lam * np.maximum(ranks - k_reg, 0)

    scores = np.empty_like(probs)
    np.put_along_axis(scores, order, cumsum, axis=1)
    return scores


SCORE_FNS = {
    "lac": scores_lac,
    "aps": scores_aps,
    "raps": scores_raps,
}