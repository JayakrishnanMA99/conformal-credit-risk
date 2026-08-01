"""Day 6: break exchangeability, then repair it with weighted conformal."""

import time
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
from src.shift import (
    make_probabilistic_shift_split,
    fit_domain_classifier,
    weighted_conformal_quantile,
    AGE_COL,
)

_t0 = time.time()

ALPHA = 0.1
N_SEEDS = 20

SCENARIOS = {
    "age":   dict(col=AGE_COL, cut=40, p_high=0.90, p_low=0.10),
    "limit": dict(col=0, cut=50_000, p_high=0.10, p_low=0.90),
}

X, y, _ = load_dataset("credit")
res = {}

for seed in range(N_SEEDS):
    # reference: no shift at all
    s = make_split(y, seed=seed)
    p = fit_predict("lightgbm", X, y, s, seed=seed)
    q = conformal_quantile(true_label_scores(p[s["cal"]], y[s["cal"]]), ALPHA)
    res.setdefault("iid", []).append(
        evaluate(build_sets(p[s["test"]], q), y[s["test"]])
    )

    for name, kw in SCENARIOS.items():
        s2 = make_probabilistic_shift_split(X, y, seed=seed, **kw)
        p2 = fit_predict("lightgbm", X, y, s2, seed=seed)

        pc, yc = p2[s2["cal"]], y[s2["cal"]]
        pt, yt = p2[s2["test"]], y[s2["test"]]
        cal_scores = true_label_scores(pc, yc)

        # unweighted: ignores the shift
        q_un = conformal_quantile(cal_scores, ALPHA)
        res.setdefault(f"{name}_unweighted", []).append(
            evaluate(build_sets(pt, q_un), yt)
        )

        # weighted: correct for it
        w_fn, _ = fit_domain_classifier(X, s2["cal"], s2["test"], seed=seed)
        w = w_fn(X[s2["cal"]])
        q_w = weighted_conformal_quantile(cal_scores, w, ALPHA)

        r = evaluate(build_sets(pt, q_w), yt)
        r["q_un"] = q_un
        r["q_w"] = q_w
        r["ess"] = float((w.sum() ** 2) / (w ** 2).sum())
        r["n_cal"] = float(len(w))
        res.setdefault(f"{name}_weighted", []).append(r)

    if (seed + 1) % 5 == 0:
        print(f"  seed {seed + 1}/{N_SEEDS}")


def stat(name, field):
    a = np.array([r[field] for r in res[name] if field in r])
    return a.mean(), a.std()


print(f"\ntarget coverage {1 - ALPHA:.2f}   ({N_SEEDS} seeds)\n")
print(f"{'setting':<20}{'coverage':>18}{'size':>10}{'gap':>10}")
print("-" * 58)
for name in res:
    cm, cs = stat(name, "coverage")
    sm, _ = stat(name, "avg_size")
    print(f"{name:<20}{cm:>10.4f} ± {cs:.4f}{sm:>10.4f}{cm - 0.9:>+10.4f}")

print(f"\n{'scenario':<10}{'paired mean diff':>18}{'sd of diff':>13}{'t':>8}{'wins':>8}")
print("-" * 57)
for name in SCENARIOS:
    un = np.array([r["coverage"] for r in res[f"{name}_unweighted"]])
    wt = np.array([r["coverage"] for r in res[f"{name}_weighted"]])

    improve = np.abs(un - 0.9) - np.abs(wt - 0.9)
    t = improve.mean() / (improve.std(ddof=1) / np.sqrt(len(improve)))
    wins = int((improve > 0).sum())
    print(f"{name:<10}{improve.mean():>+18.4f}{improve.std(ddof=1):>13.4f}"
          f"{t:>8.2f}{wins:>5d}/{len(improve)}")

print()
for name in SCENARIOS:
    e, _ = stat(f"{name}_weighted", "ess")
    n, _ = stat(f"{name}_weighted", "n_cal")
    print(f"{name:<8} n_cal {n:.0f}  ->  effective sample size {e:.0f}  ({e / n:.1%})")

print()
for name in SCENARIOS:
    qu, _ = stat(f"{name}_weighted", "q_un")
    qw, _ = stat(f"{name}_weighted", "q_w")
    print(f"{name:<8} q_hat unweighted {qu:.4f}  ->  weighted {qw:.4f}")

np.save("results/day6_weighted.npy", res, allow_pickle=True)
print(f"\nsaved -> results/day6_weighted.npy   (elapsed {time.time() - _t0:.1f}s)")