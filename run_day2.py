"""Day 2: split conformal vs calibration baselines, across 100 random splits."""

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
from src.baselines import (
    sets_from_probs,
    platt_calibrate,
    isotonic_calibrate,
)

ALPHA = 0.1
N_SEEDS = 100

X, y, _ = load_dataset("credit")
methods = {}

for seed in range(N_SEEDS):
    s = make_split(y, seed=seed)
    p = fit_predict("lightgbm", X, y, s, seed=seed)

    pc, yc = p[s["cal"]], y[s["cal"]]
    pt, yt = p[s["test"]], y[s["test"]]

    q = conformal_quantile(true_label_scores(pc, yc), ALPHA)

    run = {
        "conformal": evaluate(build_sets(pt, q), yt),
        "raw": evaluate(sets_from_probs(pt, ALPHA), yt),
        "platt": evaluate(sets_from_probs(platt_calibrate(pc, yc, pt), ALPHA), yt),
        "isotonic": evaluate(sets_from_probs(isotonic_calibrate(pc, yc, pt), ALPHA), yt),
    }

    for name, r in run.items():
        methods.setdefault(name, []).append(r)

    if (seed + 1) % 20 == 0:
        print(f"  seed {seed + 1}/{N_SEEDS}")

print(f"\ntarget coverage: {1 - ALPHA:.3f}   (n_seeds = {N_SEEDS})\n")
print(f"{'method':<12}{'coverage':>20}{'avg size':>20}{'|gap|':>10}")
print("-" * 62)

for name, runs in methods.items():
    cov = np.array([r["coverage"] for r in runs])
    size = np.array([r["avg_size"] for r in runs])
    gap = abs(cov.mean() - (1 - ALPHA))
    print(
        f"{name:<12}"
        f"{cov.mean():>12.4f} ± {cov.std():.4f}"
        f"{size.mean():>12.4f} ± {size.std():.4f}"
        f"{gap:>10.4f}"
    )

print(f"\n{'method':<12}{'singleton':>12}{'full':>12}")
print("-" * 36)
for name, runs in methods.items():
    single = np.mean([r["frac_singleton"] for r in runs])
    full = np.mean([r["frac_full"] for r in runs])
    print(f"{name:<12}{single:>12.4f}{full:>12.4f}")

np.save("results/day2_methods.npy", methods, allow_pickle=True)
print("\nsaved -> results/day2_methods.npy")