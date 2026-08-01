"""Day 5: does a learned difficulty model produce better sets?"""

import numpy as np

from src.data import load_dataset
from src.splits import make_split
from src.models import fit_predict
from src.conformal import (
    scores_lac,
    scores_aps,
    conformal_quantile,
    evaluate,
    evaluate_by_class,
    mondrian_quantiles,
    build_sets_mondrian,
)
from src.difficulty import (
    oof_true_label_scores,
    fit_difficulty,
    make_normalized_score_fn,
)

ALPHA = 0.1
N_SEEDS = 20
MODEL = "lightgbm"

X, y, _ = load_dataset("credit")
res = {}

for seed in range(N_SEEDS):
    s = make_split(y, seed=seed)
    yc, yt = y[s["cal"]], y[s["test"]]

    p = fit_predict(MODEL, X, y, s, seed=seed)
    pc, pt = p[s["cal"]], p[s["test"]]

    oof = oof_true_label_scores(MODEL, X, y, s["train"], seed=seed)
    sigma, _ = fit_difficulty(X, oof, s["train"], seed=seed)
    sig_c, sig_t = sigma(X[s["cal"]]), sigma(X[s["test"]])

    rng_c = np.random.default_rng(seed)
    rng_t = np.random.default_rng(seed + 10_000)

    variants = {
        "lac": (scores_lac, scores_lac),
        "aps": (
            lambda pr: scores_aps(pr, rng=np.random.default_rng(seed)),
            lambda pr: scores_aps(pr, rng=np.random.default_rng(seed + 10_000)),
        ),
        "lac_norm": (
            make_normalized_score_fn(sig_c, scores_lac),
            make_normalized_score_fn(sig_t, scores_lac),
        ),
    }

    for name, (fn_cal, fn_test) in variants.items():
        # marginal
        cal_sc = fn_cal(pc)[np.arange(len(yc)), yc]
        q = conformal_quantile(cal_sc, ALPHA)
        st = fn_test(pt) <= q
        r = evaluate(st, yt)
        r["cov_class1"] = evaluate_by_class(st, yt)[1]["coverage"]
        r["size_std"] = float(st.sum(axis=1).std())
        res.setdefault((name, "marginal"), []).append(r)

        # mondrian
        qs = mondrian_quantiles(pc, yc, ALPHA, score_fn=fn_cal)
        st = build_sets_mondrian(pt, qs, score_fn=fn_test)
        r = evaluate(st, yt)
        r["cov_class1"] = evaluate_by_class(st, yt)[1]["coverage"]
        r["size_std"] = float(st.sum(axis=1).std())
        res.setdefault((name, "mondrian"), []).append(r)

    print(f"  seed {seed + 1}/{N_SEEDS}")


def stat(key, field):
    a = np.array([r[field] for r in res[key]])
    return a.mean(), a.std()


for method in ["marginal", "mondrian"]:
    print(f"\n=== {method.upper()}  (target {1 - ALPHA:.2f}, {MODEL}) ===")
    print(f"{'score':<12}{'coverage':>18}{'size':>10}{'cov cls1':>11}{'singleton':>11}")
    print("-" * 62)
    for name in ["lac", "aps", "lac_norm"]:
        k = (name, method)
        cm, cs = stat(k, "coverage")
        sm, _ = stat(k, "avg_size")
        c1, _ = stat(k, "cov_class1")
        sg, _ = stat(k, "frac_singleton")
        print(f"{name:<12}{cm:>10.4f} ± {cs:.4f}{sm:>10.4f}{c1:>11.4f}{sg:>11.4f}")

np.save("results/day5_normalized.npy", res, allow_pickle=True)
print("\nsaved -> results/day5_normalized.npy")