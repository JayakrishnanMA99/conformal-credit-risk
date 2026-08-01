"""Day 6, part 1: break exchangeability and watch coverage fail."""

import numpy as np

from src.data import load_dataset
from src.splits import make_split
from src.models import fit_predict
from src.conformal import (
    true_label_scores,
    conformal_quantile,
    build_sets,
    evaluate,
)
from src.shift import make_shift_split, AGE_COL

ALPHA = 0.1
N_SEEDS = 20

X, y, _ = load_dataset("credit")

young = X[:, AGE_COL] < 40
print(f"under 40: {young.sum()}  (default rate {y[young].mean():.4f})")
print(f"40+     : {(~young).sum()}  (default rate {y[~young].mean():.4f})\n")

res = {"iid": [], "shift": []}

for seed in range(N_SEEDS):
    # baseline: normal random split
    s = make_split(y, seed=seed)
    p = fit_predict("lightgbm", X, y, s, seed=seed)
    q = conformal_quantile(true_label_scores(p[s["cal"]], y[s["cal"]]), ALPHA)
    r = evaluate(build_sets(p[s["test"]], q), y[s["test"]])
    r["q_hat"] = q
    res["iid"].append(r)

    # shifted: calibrate on young, test on old
    s2 = make_shift_split(X, y, seed=seed)
    p2 = fit_predict("lightgbm", X, y, s2, seed=seed)
    q2 = conformal_quantile(true_label_scores(p2[s2["cal"]], y[s2["cal"]]), ALPHA)
    r2 = evaluate(build_sets(p2[s2["test"]], q2), y[s2["test"]])
    r2["q_hat"] = q2
    res["shift"].append(r2)

    if (seed + 1) % 5 == 0:
        print(f"  seed {seed + 1}/{N_SEEDS}")


def stat(name, field):
    a = np.array([r[field] for r in res[name]])
    return a.mean(), a.std()


print(f"\n{'setting':<10}{'coverage':>18}{'size':>10}{'q_hat':>10}")
print("-" * 48)
for name in ["iid", "shift"]:
    cm, cs = stat(name, "coverage")
    sm, _ = stat(name, "avg_size")
    qm, _ = stat(name, "q_hat")
    print(f"{name:<10}{cm:>10.4f} ± {cs:.4f}{sm:>10.4f}{qm:>10.4f}")

np.save("results/day6_break.npy", res, allow_pickle=True)
print("\nsaved -> results/day6_break.npy")