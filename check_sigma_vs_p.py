import numpy as np

from src.data import load_dataset
from src.splits import make_split
from src.models import fit_predict
from src.conformal import true_label_scores
from src.difficulty import oof_true_label_scores, fit_difficulty

X, y, _ = load_dataset("credit")
s = make_split(y, seed=0)

p = fit_predict("lightgbm", X, y, s, seed=0)
oof = oof_true_label_scores("lightgbm", X, y, s["train"], seed=0)
sigma, _ = fit_difficulty(X, oof, s["train"], seed=0)

cal = s["cal"]
sig = sigma(X[cal])
p1 = p[cal, 1]
score = true_label_scores(p[cal], y[cal])

# how much of sigma is explained by p alone?
uncert = 1.0 - np.abs(2 * p1 - 1)          # 0 = confident, 1 = maximally unsure
print(f"corr(sigma, p1)            {np.corrcoef(sig, p1)[0,1]: .4f}")
print(f"corr(sigma, uncertainty)   {np.corrcoef(sig, uncert)[0,1]: .4f}")

# residual sigma after removing its linear fit on p1 and uncertainty
A = np.column_stack([np.ones_like(p1), p1, uncert])
beta, *_ = np.linalg.lstsq(A, sig, rcond=None)
resid = sig - A @ beta
print(f"R^2 of sigma on (p1, unc)  {1 - resid.var() / sig.var(): .4f}")
print(f"corr(residual, true score) {np.corrcoef(resid, score)[0,1]: .4f}")