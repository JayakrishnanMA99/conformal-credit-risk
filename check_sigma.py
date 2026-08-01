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

cal_sigma = sigma(X[s["cal"]])
cal_score = true_label_scores(p[s["cal"]], y[s["cal"]])

print(f"sigma range   {cal_sigma.min():.4f} to {cal_sigma.max():.4f}")
print(f"sigma mean    {cal_sigma.mean():.4f}   (actual mean score {cal_score.mean():.4f})")

q = np.quantile(cal_sigma, [0.2, 0.4, 0.6, 0.8, 1.0])
prev = -np.inf
print("\npredicted difficulty vs actual surprise, by quintile:")
for i, hi in enumerate(q):
    m = (cal_sigma > prev) & (cal_sigma <= hi)
    print(f"  Q{i+1}  n={m.sum():5d}  predicted {cal_sigma[m].mean():.4f}   actual {cal_score[m].mean():.4f}")
    prev = hi

corr = np.corrcoef(cal_sigma, cal_score)[0, 1]
print(f"\ncorrelation sigma vs actual score: {corr:.4f}")