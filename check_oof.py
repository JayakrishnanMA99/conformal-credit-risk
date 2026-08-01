import numpy as np

from src.data import load_dataset
from src.splits import make_split
from src.models import fit_predict
from src.conformal import true_label_scores
from src.difficulty import oof_true_label_scores

X, y, _ = load_dataset("credit")
s = make_split(y, seed=0)

p = fit_predict("lightgbm", X, y, s, seed=0)
in_sample = true_label_scores(p[s["train"]], y[s["train"]])
oof = oof_true_label_scores("lightgbm", X, y, s["train"], seed=0)
cal = true_label_scores(p[s["cal"]], y[s["cal"]])

print(f"train, in-sample   mean {in_sample.mean():.4f}   median {np.median(in_sample):.4f}")
print(f"train, out-of-fold mean {oof.mean():.4f}   median {np.median(oof):.4f}")
print(f"calibration        mean {cal.mean():.4f}   median {np.median(cal):.4f}")