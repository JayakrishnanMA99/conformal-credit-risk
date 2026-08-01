"""Which subgroup split breaks coverage hardest?"""

import numpy as np

from src.data import load_dataset
from src.models import fit_predict
from src.conformal import true_label_scores, conformal_quantile, build_sets, evaluate

ALPHA = 0.1
N_SEEDS = 5

X, y, _ = load_dataset("credit")

# (name, column, rule) -- rule returns True for the CALIBRATION group
cands = [
    ("age < 40",        4,  lambda v: v < 40),
    ("age < 30",        4,  lambda v: v < 30),
    ("limit < 100k",    0,  lambda v: v < 100_000),
    ("limit < 50k",     0,  lambda v: v < 50_000),
    ("education 1-2",   2,  lambda v: v <= 2),
    ("sex == 2",        1,  lambda v: v == 2),
]

print(f"{'split':<16}{'n_cal_grp':>10}{'n_test_grp':>11}{'dr_cal':>9}{'dr_test':>9}{'coverage':>11}")
print("-" * 66)

for name, col, rule in cands:
    grp = rule(X[:, col])
    a, b = np.where(grp)[0], np.where(~grp)[0]
    if len(a) < 3000 or len(b) < 3000:
        print(f"{name:<16}  skipped (group too small)")
        continue

    covs = []
    for seed in range(N_SEEDS):
        rng = np.random.default_rng(seed)
        a_sh = rng.permutation(a)
        n_tr = int(0.6 * len(a_sh))
        s = {"train": a_sh[:n_tr], "cal": a_sh[n_tr:], "test": b}

        p = fit_predict("lightgbm", X, y, s, seed=seed)
        q = conformal_quantile(true_label_scores(p[s["cal"]], y[s["cal"]]), ALPHA)
        covs.append(evaluate(build_sets(p[s["test"]], q), y[s["test"]])["coverage"])

    print(
        f"{name:<16}{len(a):>10}{len(b):>11}"
        f"{y[a].mean():>9.4f}{y[b].mean():>9.4f}{np.mean(covs):>11.4f}"
    )