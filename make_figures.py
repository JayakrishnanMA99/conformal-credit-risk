"""Day 7: figures from saved results."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data import load_dataset
from src.splits import make_split
from src.models import fit_predict
from src.conformal import (
    true_label_scores, conformal_quantile, build_sets,
    mondrian_quantiles, build_sets_mondrian,
)
from src.calibration_metrics import reliability_curve

plt.rcParams.update({
    "figure.dpi": 130,
    "font.size": 9,
    "axes.grid": True,
    "grid.alpha": 0.3,
})

TARGET = 0.9

# ---------- Figure 1: coverage by method ----------
m = np.load("results/day2_methods.npy", allow_pickle=True).item()
names = ["conformal", "raw", "platt", "isotonic"]
labels = ["Conformal", "Raw threshold", "Platt", "Isotonic"]

cov = [np.array([r["coverage"] for r in m[n]]) for n in names]
size = [np.array([r["avg_size"] for r in m[n]]) for n in names]

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))

ax[0].axhline(TARGET, color="crimson", ls="--", lw=1.2, label="Target (0.90)")
ax[0].boxplot(cov, tick_labels=labels, widths=0.55)
ax[0].set_ylabel("Empirical coverage")
ax[0].set_title("Coverage over 100 random splits")
ax[0].legend(fontsize=8)
ax[0].tick_params(axis="x", rotation=15)

ax[1].boxplot(size, tick_labels=labels, widths=0.55)
ax[1].set_ylabel("Average set size")
ax[1].set_title("Efficiency (lower is better)")
ax[1].tick_params(axis="x", rotation=15)

fig.tight_layout()
fig.savefig("figures/fig1_coverage_vs_baselines.png", bbox_inches="tight")
plt.close(fig)
print("fig1 done")

# ---------- Figure 2: per-class coverage ----------
d3 = np.load("results/day3_mondrian.npy", allow_pickle=True).item()

fig, ax = plt.subplots(figsize=(5.4, 3.4))
x = np.arange(2)
w = 0.35

for i, (key, lab, c) in enumerate([
    ("marginal", "Marginal conformal", "#4C72B0"),
    ("mondrian", "Mondrian (class-conditional)", "#DD8452"),
]):
    means = [np.mean([r[f"cov_class{k}"] for r in d3[key]]) for k in (0, 1)]
    errs = [np.std([r[f"cov_class{k}"] for r in d3[key]]) for k in (0, 1)]
    ax.bar(x + (i - 0.5) * w, means, w, yerr=errs, capsize=3, label=lab, color=c)

ax.axhline(TARGET, color="crimson", ls="--", lw=1.2, label="Target (0.90)")
ax.set_xticks(x)
ax.set_xticklabels(["Class 0 (repays)", "Class 1 (defaults)"])
ax.set_ylabel("Coverage")
ax.set_ylim(0, 1.05)
ax.set_title("Marginal coverage hides minority-class failure")
ax.legend(fontsize=8, loc="lower right")

fig.tight_layout()
fig.savefig("figures/fig2_class_conditional.png", bbox_inches="tight")
plt.close(fig)
print("fig2 done")

# ---------- Figure 3: set-size distribution ----------
X, y, _ = load_dataset("credit")
s = make_split(y, seed=0)
p = fit_predict("lightgbm", X, y, s, seed=0)
pc, yc, pt, yt = p[s["cal"]], y[s["cal"]], p[s["test"]], y[s["test"]]

q = conformal_quantile(true_label_scores(pc, yc), 1 - TARGET)
sz_marg = build_sets(pt, q).sum(axis=1)
qs = mondrian_quantiles(pc, yc, 1 - TARGET)
sz_mond = build_sets_mondrian(pt, qs).sum(axis=1)

fig, ax = plt.subplots(figsize=(5.4, 3.4))
vals = [0, 1, 2]
w = 0.35
for i, (sz, lab, c) in enumerate([
    (sz_marg, "Marginal", "#4C72B0"),
    (sz_mond, "Mondrian", "#DD8452"),
]):
    fr = [(sz == v).mean() for v in vals]
    ax.bar(np.array(vals) + (i - 0.5) * w, fr, w, label=lab, color=c)

ax.set_xticks(vals)
ax.set_xticklabels(["Empty", "Singleton\n(auto-decide)", "Both labels\n(human review)"])
ax.set_ylabel("Fraction of test clients")
ax.set_title("The cost of class-conditional coverage")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig("figures/fig3_set_sizes.png", bbox_inches="tight")
plt.close(fig)
print("fig3 done")

# ---------- Figure 4: reliability diagram ----------
conf, acc, cnt = reliability_curve(pt[:, 1], yt, n_bins=10)
ok = cnt > 0

fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))

ax[0].plot([0, 1], [0, 1], color="crimson", ls="--", lw=1.2, label="Perfect calibration")
ax[0].plot(conf[ok], acc[ok], "o-", color="#4C72B0", label="LightGBM")
ax[0].set_xlabel("Mean predicted P(default)")
ax[0].set_ylabel("Observed default rate")
ax[0].set_title("Reliability diagram")
ax[0].legend(fontsize=8)

ax[1].bar(np.arange(10)[ok], cnt[ok], color="#4C72B0")
ax[1].set_xlabel("Probability bin")
ax[1].set_ylabel("Number of clients")
ax[1].set_title("Most clients sit in low-probability bins")

fig.tight_layout()
fig.savefig("figures/fig4_reliability.png", bbox_inches="tight")
plt.close(fig)
print("fig4 done")

print("\nall figures -> figures/")