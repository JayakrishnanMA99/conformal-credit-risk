"""Day 3: marginal vs class-conditional (Mondrian) conformal."""

import numpy as np

from src.data import load_dataset
from src.splits import make_split
from src.models import fit_predict
from src.conformal import (
    true_label_scores,
    conformal_quantile,
    build_sets,
    evaluate,
    evaluate_by_class,
    mondrian_quantiles,
    build_sets_mondrian,
)

ALPHA = 0.1
N_SEEDS = 100

X, y, _ = load_dataset("credit")
res = {"marginal": [], "mondrian": []}

for seed in range(N_SEEDS):
    s = make_split(y, seed=seed)
    p = fit_predict("lightgbm", X, y, s, seed=seed)

    pc, yc = p[s["cal"]], y[s["cal"]]
    pt, yt = p[s["test"]], y[s["test"]]

    q = conformal_quantile(true_label_scores(pc, yc), ALPHA)
    sets_marg = build_sets(pt, q)

    qs = mondrian_quantiles(pc, yc, ALPHA)
    sets_mond = build_sets_mondrian(pt, qs)

    for name, st in [("marginal", sets_marg), ("mondrian", sets_mond)]:
        r = evaluate(st, yt)
        by_c = evaluate_by_class(st, yt)
        r["cov_class0"] = by_c[0]["coverage"]
        r["cov_class1"] = by_c[1]["coverage"]
        res[name].append(r)

    if (seed + 1) % 20 == 0:
        print(f"  seed {seed + 1}/{N_SEEDS}")


def col(name, key):
    return np.array([r[key] for r in res[name]])


print(f"\ntarget coverage: {1 - ALPHA:.3f}   (n_seeds = {N_SEEDS})\n")
print(f"{'method':<12}{'overall':>18}{'class 0':>18}{'class 1':>18}")
print("-" * 66)
for name in res:
    print(
        f"{name:<12}"
        f"{col(name,'coverage').mean():>10.4f} ± {col(name,'coverage').std():.4f}"
        f"{col(name,'cov_class0').mean():>10.4f} ± {col(name,'cov_class0').std():.4f}"
        f"{col(name,'cov_class1').mean():>10.4f} ± {col(name,'cov_class1').std():.4f}"
    )

print(f"\n{'method':<12}{'avg size':>18}{'singleton':>12}")
print("-" * 42)
for name in res:
    print(
        f"{name:<12}"
        f"{col(name,'avg_size').mean():>10.4f} ± {col(name,'avg_size').std():.4f}"
        f"{col(name,'frac_singleton').mean():>12.4f}"
    )

np.save("results/day3_mondrian.npy", res, allow_pickle=True)
print("\nsaved -> results/day3_mondrian.npy")