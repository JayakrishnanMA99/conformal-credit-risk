"""Day 4: does validity hold across base models and score functions?"""

import numpy as np

from src.data import load_dataset
from src.splits import make_split
from src.models import fit_predict, MODELS
from src.conformal import (
    conformal_quantile,
    evaluate,
    evaluate_by_class,
    mondrian_quantiles,
    build_sets_mondrian,
    SCORE_FNS,
)

ALPHA = 0.1
N_SEEDS = 30

X, y, _ = load_dataset("credit")
res = {}

for seed in range(N_SEEDS):
    s = make_split(y, seed=seed)
    yc, yt = y[s["cal"]], y[s["test"]]

    for model_name in MODELS:
        p = fit_predict(model_name, X, y, s, seed=seed)
        pc, pt = p[s["cal"]], p[s["test"]]

        for score_name, score_fn in SCORE_FNS.items():
            # fresh generator per (seed, model, score) so randomized
            # scores vary across seeds but stay reproducible
            def fn(probs, _f=score_fn, _s=seed):
                try:
                    return _f(probs, rng=np.random.default_rng(_s))
                except TypeError:
                    return _f(probs)

            # marginal: one shared threshold
            cal_scores = fn(pc)[np.arange(len(yc)), yc]
            q = conformal_quantile(cal_scores, ALPHA)
            sets_marg = fn(pt) <= q

            # mondrian: one threshold per class
            qs = mondrian_quantiles(pc, yc, ALPHA, score_fn=fn)
            sets_mond = build_sets_mondrian(pt, qs, score_fn=fn)

            for method, st in [("marginal", sets_marg), ("mondrian", sets_mond)]:
                r = evaluate(st, yt)
                by_c = evaluate_by_class(st, yt)
                r["cov_class1"] = by_c[1]["coverage"]
                res.setdefault((model_name, score_name, method), []).append(r)

    print(f"  seed {seed + 1}/{N_SEEDS}")


def stat(key, field):
    a = np.array([r[field] for r in res[key]])
    return a.mean(), a.std()


for method in ["marginal", "mondrian"]:
    print(f"\n=== {method.upper()}  (target {1 - ALPHA:.2f}) ===")
    print(f"{'model':<15}{'score':<8}{'coverage':>18}{'size':>10}{'cov cls1':>11}")
    print("-" * 62)
    for model_name in MODELS:
        for score_name in SCORE_FNS:
            k = (model_name, score_name, method)
            cm, cs = stat(k, "coverage")
            sm, _ = stat(k, "avg_size")
            c1m, _ = stat(k, "cov_class1")
            print(f"{model_name:<15}{score_name:<8}{cm:>10.4f} ± {cs:.4f}{sm:>10.4f}{c1m:>11.4f}")

np.save("results/day4_grid.npy", res, allow_pickle=True)
print("\nsaved -> results/day4_grid.npy")