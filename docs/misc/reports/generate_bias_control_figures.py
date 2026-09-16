#!/usr/bin/env python3
"""
Generates the bias-control figures for the multisteering report: the
digit-logit contrast for counting, and the sham/injected/double-adjusted
"reveal" comparison for ordering. Companion to generate_figures.py (which
covers the original, pre-bias-control report) -- run from the repo root:

    .venv/bin/python docs/misc/reports/generate_bias_control_figures.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import ttest_1samp

REPO = Path(__file__).resolve().parents[3]
RESULTS = REPO / "results" / "multisteering"
PLOT_DIRS = [
    RESULTS / "experiment_9_counting" / "raw",
    RESULTS / "experiment_10_identification" / "raw",
    RESULTS / "experiment_11_ordering" / "raw",
    RESULTS / "experiment_11_modulators" / "raw",
]
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)


def load(name):
    for directory in PLOT_DIRS:
        path = directory / name
        if path.exists():
            return torch.load(path, weights_only=False)
    searched = ", ".join(str(directory / name) for directory in PLOT_DIRS)
    raise FileNotFoundError(f"Could not find {name}; searched: {searched}")


def savefig(fig, name):
    fig.tight_layout()
    fig.savefig(FIGDIR / name, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {name}")


def mean_ci(vals, z=1.96):
    vals = np.asarray(vals, dtype=float)
    mean = vals.mean()
    sem = vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0
    return mean, z * sem


# ============================================================
# 1. Counting: digit-logit contrast by dose (individual regime)
# ============================================================
d = load("multi_detection_trials_individual.pt")
trials = [t for t in d["trials"] if t["prompt_style"] == "count" and t["k_actual"] != 1]
alphas = sorted(set(t["alpha"] for t in trials))

fig, ax = plt.subplots(figsize=(8, 5))
width = 0.35
x = np.arange(len(alphas))
for i, condition in enumerate(["real", "random"]):
    means, errs = [], []
    for a in alphas:
        vals = [t["logit_contrast_adjusted"] for t in trials if t["alpha"] == a and t["condition"] == condition]
        m, e = mean_ci(vals)
        means.append(m)
        errs.append(e)
    ax.bar(x + (i - 0.5) * width, means, width, yerr=errs, capsize=4, label=condition)
ax.axhline(0, color="gray", linewidth=0.8)
ax.set_xticks(x)
ax.set_xticklabels([str(a) for a in alphas])
ax.set_xlabel("alpha")
ax.set_ylabel("mean logit(true k) - logit(\"1\")  [95% CI]")
ax.set_title("Counting: true-count digit is significantly LESS supported\nthan the default \"1\", at every dose")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")
savefig(fig, "summary_counting_digit_logit_contrast.png")

all_vals = [t["logit_contrast_adjusted"] for t in trials]
t_res = ttest_1samp(all_vals, 0.0)
print(f"  counting: overall mean={np.mean(all_vals):+.3f}, n={len(all_vals)}, "
      f"t={t_res.statistic:.2f}, p={t_res.pvalue:.2e}")


# ============================================================
# 2. Ordering: injected vs. sham vs. double-adjusted, overall + by alpha
# ============================================================
d = load("layer_ordering_trials.pt")
trials = d["trials"]
alphas = sorted(set(t["alpha"] for t in trials))

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Panel 1: overall bars
labels = ["injected", "sham (no injection)", "double-adjusted\n(injected - sham)"]
fields = ["logit_contrast_adjusted", "logit_contrast_adjusted_sham", "logit_contrast_double_adjusted"]
means, errs = [], []
for f in fields:
    vals = [t[f] for t in trials]
    m, e = mean_ci(vals)
    means.append(m)
    errs.append(e)
colors = ["steelblue", "gray", "indianred"]
bars = axes[0].bar(labels, means, yerr=errs, capsize=5, color=colors)
axes[0].axhline(0, color="gray", linewidth=0.8)
axes[0].set_ylabel("mean logit(correct letter) - logit(incorrect letter)  [95% CI]")
axes[0].set_title("Ordering: the injected effect is\nindistinguishable from the sham baseline")
axes[0].grid(True, alpha=0.3, axis="y")

# Panel 2: by alpha, three lines
for f, label, color in zip(fields, labels, colors):
    line_means = []
    for a in alphas:
        vals = [t[f] for t in trials if t["alpha"] == a]
        line_means.append(np.mean(vals))
    axes[1].plot(alphas, line_means, marker="o", label=label.replace("\n", " "), color=color)
axes[1].axhline(0, color="gray", linewidth=0.8)
axes[1].set_xlabel("alpha")
axes[1].set_ylabel("mean logit contrast")
axes[1].set_title("Same comparison, by alpha")
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.3)

fig.suptitle("Paired-sham control reveals the apparent ordering signal is a prompt-level baseline, not injection-driven")
savefig(fig, "summary_ordering_sham_reveal.png")

for f, label in zip(fields, ["injected", "sham", "double-adjusted"]):
    vals = [t[f] for t in trials]
    t_res = ttest_1samp(vals, 0.0)
    print(f"  ordering {label}: mean={np.mean(vals):+.3f}, n={len(vals)}, "
          f"t={t_res.statistic:.3f}, p={t_res.pvalue:.3g}")

print("\nAll bias-control figures saved to", FIGDIR)
