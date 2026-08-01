import numpy as np
from sklearn.metrics import roc_auc_score

from src.data import load_dataset
from src.shift import fit_domain_classifier, make_probabilistic_shift_split, AGE_COL

X, y, _ = load_dataset("credit")

for name, kw in [
    ("age soft",   dict(col=AGE_COL, cut=40)),
    ("limit soft", dict(col=0, cut=50_000, p_high=0.25, p_low=0.75)),
]:
    s = make_probabilistic_shift_split(X, y, seed=0, **kw)
    cal, test = s["cal"], s["test"]

    w_fn, m = fit_domain_classifier(X, cal, test, seed=0)

    Xd = np.vstack([X[cal], X[test]])
    yd = np.concatenate([np.zeros(len(cal)), np.ones(len(test))])
    auc = roc_auc_score(yd, m.predict_proba(Xd)[:, 1])

    w = w_fn(X[cal])
    print(f"{name:<12} n_cal {len(cal):5d}  n_test {len(test):5d}  domain AUC {auc:.4f}")
    print(f"             weights: min {w.min():.3f}  median {np.median(w):.3f}  "
          f"max {w.max():.3f}  frac clipped {np.mean((w <= 0.0501) | (w >= 19.99)):.3f}\n")