#!/usr/bin/env python3
"""
Generates every figure used in multisteering_report.md from the saved trial
files in results/multisteering/, and prints the numeric tables the report's
prose and markdown tables are built from. Run from the repo root:

    .venv/bin/python docs/misc/reports/generate_figures.py

Not part of the experiment pipeline proper (code/experiments, code/analysis)
-- this is report-specific plotting, kept alongside the report so the
figures can be regenerated after future re-runs.
"""

import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.stats import binomtest, ttest_1samp

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "code" / "utils"))

RESULTS = REPO / "results" / "multisteering"
PLOT_DIRS = [
    RESULTS / "experiment_9_counting" / "raw",
    RESULTS / "experiment_10_identification" / "raw",
    RESULTS / "experiment_11_ordering" / "raw",
    RESULTS / "experiment_11_modulators" / "raw",
]
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

CHANCE_KW = dict(color="gray", linestyle=":", linewidth=1, label="chance")


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


# ============================================================
# Experience 9 -- counting
# ============================================================
print("=" * 70)
print("EXPERIENCE 9 -- COUNTING")
print("=" * 70)

d_ind = load("multi_detection_trials_individual.pt")
trials9 = d_ind["trials"]

# Response distribution by actual k, condition=real, aggregated over alpha.
fig, ax = plt.subplots(figsize=(8, 5))
k_values = [0, 1, 2, 3, 4]
answers = [0, 1, 2, 3, 4, None]
width = 0.14
x = np.arange(len(k_values))
for i, ans in enumerate(answers):
    fracs = []
    for k in k_values:
        sub = [t for t in trials9 if t["condition"] == "real" and t["k_actual"] == k
               and t["prompt_style"] == "count"]
        n = len(sub)
        frac = sum(1 for t in sub if t["reported_count"] == ans) / n if n else 0
        fracs.append(frac)
    label = str(ans) if ans is not None else "unparsed"
    ax.bar(x + (i - 2.5) * width, fracs, width, label=label)
ax.set_xticks(x)
ax.set_xticklabels([f"k={k}" for k in k_values])
ax.set_xlabel("actual number of injections")
ax.set_ylabel("fraction of trials reporting each count")
ax.set_title("Exp 9: reported count vs. actual k\n(condition=real, all alphas pooled)")
ax.legend(title="reported", fontsize=8, ncol=3)
ax.grid(True, alpha=0.3, axis="y")
savefig(fig, "e9_response_distribution.png")

# Detection rate (reported>0) by alpha, k=0 vs k=4, both conditions.
fig, ax = plt.subplots(figsize=(8, 5))
for condition in ["real", "random"]:
    for k, style in [(0, "--"), (4, "-")]:
        alphas = sorted(set(t["alpha"] for t in trials9))
        rates = []
        for a in alphas:
            sub = [t for t in trials9 if t["condition"] == condition and t["k_actual"] == k
                   and t["alpha"] == a and t["prompt_style"] == "count" and t["is_coherent"]]
            n = len(sub)
            rate = sum(1 for t in sub if t["reported_count"] and t["reported_count"] > 0) / n if n else float("nan")
            rates.append(rate)
        ax.plot(alphas, rates, style, marker="o", label=f"{condition}, k={k}")
ax.set_xlabel("alpha")
ax.set_ylabel("fraction reporting count > 0")
ax.set_ylim(0, 1.05)
ax.set_title("Exp 9: 'detects >0' rate at k=0 (false positive) vs. k=4 (should differ)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
savefig(fig, "e9_detection_rate_by_alpha.png")

fp_rate_k0 = np.mean([1 for t in trials9 if t["k_actual"] == 0 and t["prompt_style"] == "count"
                       and t["is_coherent"] and t["reported_count"] == 1])
print(f"  fraction of k=0 trials reporting exactly '1': {fp_rate_k0:.1%}")


# ============================================================
# Experience 10 -- identification
# ============================================================
print("\n" + "=" * 70)
print("EXPERIENCE 10 -- IDENTIFICATION")
print("=" * 70)

CONDITIONS10 = ["single_concept", "two_concepts", "concept_plus_random", "sham"]
d10 = {c: load(f"multi_identification_trials_{c}.pt") for c in CONDITIONS10}


def exact_match_rate(trials):
    coherent = [t for t in trials if t["mode"] == "forced_choice" and t["is_coherent"]]
    if not coherent:
        return float("nan"), 0
    from collections import Counter
    matches = [Counter(t["reported_labels"]) == Counter(t["correct_labels"]) for t in coherent]
    return float(np.mean(matches)), len(coherent)


def free_frac_above(trials, threshold=0.5):
    coherent = [t for t in trials if t["mode"] == "free" and t["is_coherent"] and t.get("sim_to_injected")]
    if not coherent:
        return float("nan"), 0
    sims = [s for t in coherent for s in t["sim_to_injected"]]
    return float(np.mean([s > threshold for s in sims])), len(sims)


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
em_rates, em_ns = [], []
fa_rates, fa_ns = [], []
for c in CONDITIONS10:
    em, n_em = exact_match_rate(d10[c]["trials"])
    fa, n_fa = free_frac_above(d10[c]["trials"])
    em_rates.append(em)
    em_ns.append(n_em)
    fa_rates.append(fa)
    fa_ns.append(n_fa)

axes[0].bar(CONDITIONS10, em_rates, color="steelblue")
axes[0].set_ylabel("exact-match accuracy (forced choice)")
axes[0].set_ylim(0, max(0.05, max(v for v in em_rates if v == v) * 1.5))
axes[0].set_title("Forced-choice exact match, both labels correct")
axes[0].tick_params(axis="x", rotation=20)
axes[0].grid(True, alpha=0.3, axis="y")

axes[1].bar(CONDITIONS10, fa_rates, color="darkorange")
axes[1].axhline(0, color="gray", linewidth=0.5)
axes[1].set_ylabel("fraction of concept-scores > 0.5 similarity")
axes[1].set_title("Free response, similarity to true injected concept")
axes[1].tick_params(axis="x", rotation=20)
axes[1].grid(True, alpha=0.3, axis="y")

fig.suptitle("Exp 10: identification metrics by condition (all alphas pooled)")
savefig(fig, "e10_metrics_by_condition.png")

# By alpha, two_concepts (the richest condition: 2 real concept scores/trial)
fig, ax = plt.subplots(figsize=(8, 5))
alphas10 = sorted(set(t["alpha"] for t in d10["two_concepts"]["trials"]))
em_by_a, fa_by_a, fbd_by_a = [], [], []
for a in alphas10:
    sub = [t for t in d10["two_concepts"]["trials"] if t["alpha"] == a]
    em, _ = exact_match_rate(sub)
    fa, _ = free_frac_above(sub)
    coherent = [t for t in sub if t["mode"] == "free" and t["is_coherent"] and t.get("sim_to_injected")]
    fbd = np.mean([all(s > t.get("best_distractor_similarity", -1.0) for s in t["sim_to_injected"])
                    for t in coherent]) if coherent else float("nan")
    em_by_a.append(em)
    fa_by_a.append(fa)
    fbd_by_a.append(fbd)
ax.plot(alphas10, em_by_a, marker="o", label="forced-choice exact match")
ax.plot(alphas10, fa_by_a, marker="s", label="free response > 0.5 similarity")
ax.plot(alphas10, fbd_by_a, marker="^", label="free response beats best distractor")
ax.set_xlabel("alpha")
ax.set_ylabel("rate")
ax.set_ylim(0, 1)
ax.set_title("Exp 10 (two_concepts): identification metrics vs. alpha")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
savefig(fig, "e10_metrics_by_alpha.png")

for c in CONDITIONS10:
    em, n_em = exact_match_rate(d10[c]["trials"])
    print(f"  {c}: exact_match={em:.1%} (n={n_em})")


# ============================================================
# Experience 11 -- ordering
# ============================================================
print("\n" + "=" * 70)
print("EXPERIENCE 11 -- ORDERING")
print("=" * 70)

d11 = load("layer_ordering_trials.pt")
trials11 = d11["trials"]
coherent11 = [t for t in trials11 if t["is_coherent"] and t["reported_letter"] is not None]

# Label-assignment bias bar chart
fig, ax = plt.subplots(figsize=(7, 5))
by_assignment = defaultdict(list)
for t in coherent11:
    by_assignment[t["label_assignment"]].append(t["correct"])
assignments = sorted(by_assignment)
accs = [np.mean(by_assignment[a]) for a in assignments]
ns = [len(by_assignment[a]) for a in assignments]
bars = ax.bar([a.replace("_", "\n") for a in assignments], accs, color=["seagreen", "indianred"])
ax.axhline(0.5, **CHANCE_KW)
for bar, n in zip(bars, ns):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={n}", ha="center", fontsize=9)
ax.set_ylabel("accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Exp 11: accuracy by label assignment\n(reveals the 'always answer A' bias)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, axis="y")
savefig(fig, "e11_label_bias.png")

overall_acc = np.mean([t["correct"] for t in coherent11])
n_overall = len(coherent11)
binom_p = binomtest(sum(t["correct"] for t in coherent11), n_overall, p=0.5).pvalue
print(f"  overall accuracy: {overall_acc:.1%} (n={n_overall}), binomial p={binom_p:.3g}")

# Accuracy and logit contrast vs alpha, side by side
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
alphas11 = sorted(set(t["alpha"] for t in trials11))
acc_by_a = [np.mean([t["correct"] for t in coherent11 if t["alpha"] == a]) for a in alphas11]
logit_by_a = [np.mean([t["logit_contrast_adjusted"] for t in trials11 if t["alpha"] == a]) for a in alphas11]

axes[0].plot(alphas11, acc_by_a, marker="o", color="steelblue")
axes[0].axhline(0.5, **CHANCE_KW)
axes[0].set_xlabel("alpha")
axes[0].set_ylabel("accuracy (discrete letter)")
axes[0].set_ylim(0, 1)
axes[0].set_title("Raw accuracy vs. alpha\n(dominated by the A-bias)")
axes[0].grid(True, alpha=0.3)

axes[1].plot(alphas11, logit_by_a, marker="o", color="darkorange")
axes[1].axhline(0, color="gray", linewidth=0.5)
axes[1].set_xlabel("alpha")
axes[1].set_ylabel("mean logit(correct) - logit(incorrect)")
axes[1].set_title("Bias-adjusted logit contrast vs. alpha\n(all trials, incl. incoherent)")
axes[1].grid(True, alpha=0.3)

fig.suptitle("Exp 11 (layer_ordering): two views of the same data")
savefig(fig, "e11_accuracy_vs_logit_by_alpha.png")

t_all = ttest_1samp([t["logit_contrast_adjusted"] for t in trials11], 0.0)
print(f"  logit contrast (all trials): mean={np.mean([t['logit_contrast_adjusted'] for t in trials11]):+.3f}, "
      f"t={t_all.statistic:.3f}, p={t_all.pvalue:.3g}")

# Accuracy by distance
fig, ax = plt.subplots(figsize=(8, 5))
by_distance = defaultdict(list)
for t in coherent11:
    by_distance[t["layer_distance"]].append(t["correct"])
distances = sorted(by_distance)
accs_d = [np.mean(by_distance[dd]) for dd in distances]
ns_d = [len(by_distance[dd]) for dd in distances]
sizes = [20 + 4 * n for n in ns_d]
ax.scatter(distances, accs_d, s=sizes, alpha=0.7, color="steelblue")
ax.axhline(0.5, **CHANCE_KW)
ax.set_xlabel("layer distance |i-j|")
ax.set_ylabel("accuracy")
ax.set_ylim(0, 1.05)
ax.set_title("Exp 11: accuracy by layer distance (point size = n trials)")
ax.grid(True, alpha=0.3)
savefig(fig, "e11_accuracy_by_distance.png")


# ============================================================
# Modulators (E4/E5/E6)
# ============================================================
print("\n" + "=" * 70)
print("MODULATORS (E4/E5/E6)")
print("=" * 70)

d_e4 = load("modulators_trials_e4_distance.pt")["trials"]
d_e5 = load("modulators_trials_e5_alpha_ratio.pt")["trials"]
d_e6 = load("modulators_trials_e6_similarity.pt")["trials"]

# E4: logit contrast by distance, split by placement
fig, ax = plt.subplots(figsize=(8, 5))
for placement in sorted(set(t["placement"] for t in d_e4)):
    by_d = defaultdict(list)
    for t in d_e4:
        if t["placement"] == placement:
            by_d[t["distance"]].append(t["logit_contrast_adjusted"])
    ds = sorted(by_d)
    means = [np.mean(by_d[dd]) for dd in ds]
    ax.plot(ds, means, marker="o", label=placement)
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("layer distance |i-j|")
ax.set_ylabel("mean logit(correct) - logit(incorrect)")
ax.set_title("E4: bias-adjusted logit contrast vs. distance\n(significant overall: p=0.0006)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
savefig(fig, "modulators_e4_logit_by_distance.png")

e4_all = [t["logit_contrast_adjusted"] for t in d_e4]
t_e4 = ttest_1samp(e4_all, 0.0)
print(f"  E4 logit contrast: mean={np.mean(e4_all):+.3f}, n={len(e4_all)}, t={t_e4.statistic:.3f}, p={t_e4.pvalue:.3g}")

# E5 & E6: the bias-decomposition bar chart (the key explanatory figure)
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, trials, label in [(axes[0], d_e5, "E5 (dose ratio)"), (axes[1], d_e6, "E6 (similarity)")]:
    means, ns = [], []
    for correct in ["A", "B"]:
        sub = [t["logit_contrast_adjusted"] for t in trials if t["correct_letter"] == correct]
        means.append(np.mean(sub))
        ns.append(len(sub))
    bars = ax.bar(["correct=A", "correct=B"], means, color=["seagreen", "indianred"])
    span = max(means) - min(means)
    for bar, n in zip(bars, ns):
        y = bar.get_height()
        offset = 0.05 * span if y >= 0 else -0.05 * span
        va = "bottom" if y >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y + offset, f"n={n}", ha="center", va=va, fontsize=9)
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.margins(y=0.15)
    ax.set_ylabel("mean logit(correct) - logit(incorrect)")
    ax.set_title(f"{label}: split by which letter\nwas actually correct")
    ax.grid(True, alpha=0.3, axis="y")
fig.suptitle("The overall-negative E5/E6 means are the same A-bias,\nunevenly sampled -- not a new ratio/similarity effect")
savefig(fig, "modulators_e5_e6_bias_decomposition.png")

for name, trials in [("E5", d_e5), ("E6", d_e6)]:
    all_vals = [t["logit_contrast_adjusted"] for t in trials]
    t_res = ttest_1samp(all_vals, 0.0)
    n_a = sum(1 for t in trials if t["correct_letter"] == "A")
    n_b = sum(1 for t in trials if t["correct_letter"] == "B")
    print(f"  {name}: overall mean={np.mean(all_vals):+.3f} (p={t_res.pvalue:.3g}), "
          f"correct=A n={n_a}, correct=B n={n_b}")

# E5 by ratio (accuracy, for completeness alongside earlier logit view)
fig, ax = plt.subplots(figsize=(8, 5))
by_ratio = defaultdict(list)
for t in d_e5:
    if t["is_coherent"] and t["correct"] is not None:
        by_ratio[t["ratio"]].append(t["correct"])
ratios = sorted(by_ratio)
accs_r = [np.mean(by_ratio[r]) for r in ratios]
ns_r = [len(by_ratio[r]) for r in ratios]
ax.plot(ratios, accs_r, marker="o", color="steelblue")
ax.axhline(0.5, **CHANCE_KW)
ax.set_xscale("log")
ax.set_xlabel("dose ratio z_A / z_B")
ax.set_ylabel("raw accuracy")
ax.set_ylim(0, 1)
ax.set_title("E5: raw accuracy vs. dose ratio\n(likely destabilized at high ratio x alpha)")
ax.grid(True, alpha=0.3)
savefig(fig, "modulators_e5_accuracy_by_ratio.png")

# ============================================================
# Appendix -- alpha dependence across all experiments
# ============================================================
print("\n" + "=" * 70)
print("APPENDIX -- ALPHA DEPENDENCE")
print("=" * 70)

# --- A.1: Exp 9 coherence rate by dose, both regimes x both conditions ---
# individual regime's per-trial "alpha" is the swept dose directly; budget
# regime's per-trial "alpha" is the *derived* per-injection dose
# (z_total/sqrt(k)), which mixes across k for a given z_total -- so each
# regime needs its own natural x-axis (alpha vs. z_total), not a shared one.
d_budget9 = load("multi_detection_trials_budget.pt")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, regime, trials, dose_field in [
    (axes[0], "individual", d_ind["trials"], "alpha"),
    (axes[1], "budget", d_budget9["trials"], "z_total"),
]:
    doses = sorted(set(t[dose_field] for t in trials))
    for condition in ["real", "random"]:
        rates = []
        for dose in doses:
            sub = [t for t in trials if t[dose_field] == dose and t["condition"] == condition
                   and t["prompt_style"] == "count"]
            rates.append(np.mean([t["is_coherent"] for t in sub]) if sub else float("nan"))
        ax.plot(doses, rates, marker="o", label=condition)
    ax.set_xlabel(dose_field)
    ax.set_ylabel("coherence rate")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"{regime} regime")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
fig.suptitle("Exp 9: coherence rate vs. dose")
savefig(fig, "appendix_e9_coherence_by_alpha.png")

# --- A.2: Exp 9 mean absolute error |reported - actual k| by dose ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, regime, trials, dose_field in [
    (axes[0], "individual", d_ind["trials"], "alpha"),
    (axes[1], "budget", d_budget9["trials"], "z_total"),
]:
    doses = sorted(set(t[dose_field] for t in trials))
    maes = []
    for dose in doses:
        sub = [t for t in trials if t[dose_field] == dose and t["condition"] == "real"
               and t["prompt_style"] == "count" and t["reported_count"] is not None]
        errs = [abs(t["reported_count"] - t["k_actual"]) for t in sub]
        maes.append(np.mean(errs) if errs else float("nan"))
    ax.plot(doses, maes, marker="o", color="steelblue")
    ax.set_xlabel(dose_field)
    ax.set_ylabel("mean |reported count - actual k|")
    ax.set_title(f"{regime} regime, condition=real")
    ax.grid(True, alpha=0.3)
fig.suptitle("Exp 9: count error vs. dose\n(lower would mean tracking the true count better)")
savefig(fig, "appendix_e9_mae_by_alpha.png")

# --- A.3: Exp 10 exact-match rate by alpha, all 4 conditions ---
fig, ax = plt.subplots(figsize=(8, 5))
for c in CONDITIONS10:
    trials = d10[c]["trials"]
    alphas_c = sorted(set(t["alpha"] for t in trials))
    rates = []
    for a in alphas_c:
        sub = [t for t in trials if t["alpha"] == a]
        em, _ = exact_match_rate(sub)
        rates.append(em)
    ax.plot(alphas_c, rates, marker="o", label=c)
ax.set_xlabel("alpha")
ax.set_ylabel("forced-choice exact-match accuracy")
ax.set_title("Exp 10: exact-match accuracy vs. alpha, all conditions")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
savefig(fig, "appendix_e10_exact_match_by_alpha.png")

# --- A.4: Exp 10 free-response frac-above-threshold by alpha, 3 conditions ---
fig, ax = plt.subplots(figsize=(8, 5))
for c in ["single_concept", "two_concepts", "concept_plus_random"]:
    trials = d10[c]["trials"]
    alphas_c = sorted(set(t["alpha"] for t in trials))
    rates = []
    for a in alphas_c:
        sub = [t for t in trials if t["alpha"] == a]
        fa, _ = free_frac_above(sub)
        rates.append(fa)
    ax.plot(alphas_c, rates, marker="o", label=c)
ax.set_xlabel("alpha")
ax.set_ylabel("free response: fraction > 0.5 similarity to true concept")
ax.set_title("Exp 10: free-response identification vs. alpha")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
savefig(fig, "appendix_e10_free_response_by_alpha.png")

# --- A.5: Exp 10 coherence rate by alpha, by mode, aggregated across conditions ---
fig, ax = plt.subplots(figsize=(8, 5))
all10_trials = [t for c in CONDITIONS10 for t in d10[c]["trials"]]
alphas10_all = sorted(set(t["alpha"] for t in all10_trials))
for mode in ["forced_choice", "free"]:
    rates = []
    for a in alphas10_all:
        sub = [t for t in all10_trials if t["alpha"] == a and t["mode"] == mode]
        rates.append(np.mean([t["is_coherent"] for t in sub]) if sub else float("nan"))
    ax.plot(alphas10_all, rates, marker="o", label=mode)
ax.set_xlabel("alpha")
ax.set_ylabel("coherence rate")
ax.set_ylim(0, 1.05)
ax.set_title("Exp 10: coherence rate vs. alpha, by response format\n(all 4 conditions pooled)")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
savefig(fig, "appendix_e10_coherence_by_alpha.png")

# --- A.6: Exp 11 (layer_ordering) bias strength + coherence vs alpha ---
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
picked_a_by_a, coh_by_a = [], []
for a in alphas11:
    sub_all = [t for t in trials11 if t["alpha"] == a]
    sub_coh = [t for t in sub_all if t["is_coherent"] and t["reported_letter"] is not None]
    picked_a_by_a.append(np.mean([t["reported_letter"] == "A" for t in sub_coh]) if sub_coh else float("nan"))
    coh_by_a.append(np.mean([t["is_coherent"] for t in sub_all]))
axes[0].plot(alphas11, picked_a_by_a, marker="o", color="indianred")
axes[0].axhline(0.5, **CHANCE_KW)
axes[0].set_xlabel("alpha")
axes[0].set_ylabel("fraction answering 'A'")
axes[0].set_ylim(0, 1.05)
axes[0].set_title("'A' bias strength vs. alpha")
axes[0].legend(fontsize=8)
axes[0].grid(True, alpha=0.3)

axes[1].plot(alphas11, coh_by_a, marker="o", color="teal")
axes[1].set_xlabel("alpha")
axes[1].set_ylabel("coherence rate")
axes[1].set_ylim(0, 1.05)
axes[1].set_title("Coherence rate vs. alpha")
axes[1].grid(True, alpha=0.3)
fig.suptitle("Exp 11 (layer_ordering)")
savefig(fig, "appendix_e11_bias_and_coherence_by_alpha.png")

# --- A.7: modulators raw accuracy vs alpha, E4 / E5(ratio=1) / E6 ---
# E5 records store the swept base dose as "base_alpha" (alpha_a/alpha_b are
# derived from it via the ratio), not a plain "alpha" key like E4/E6.
def _alpha_of(t):
    return t["base_alpha"] if "base_alpha" in t else t["alpha"]


fig, ax = plt.subplots(figsize=(8, 5))
for label, trials, filt in [
    ("E4 (distance)", d_e4, lambda t: True),
    ("E5 (ratio=1, i.e. no dose asymmetry)", d_e5, lambda t: t["ratio"] == 1.0),
    ("E6 (similarity)", d_e6, lambda t: True),
]:
    sub_all = [t for t in trials if filt(t)]
    alphas_m = sorted(set(_alpha_of(t) for t in sub_all))
    accs = []
    for a in alphas_m:
        sub = [t for t in sub_all if _alpha_of(t) == a and t["is_coherent"] and t["correct"] is not None]
        accs.append(np.mean([t["correct"] for t in sub]) if sub else float("nan"))
    ax.plot(alphas_m, accs, marker="o", label=label)
ax.axhline(0.5, **CHANCE_KW)
ax.set_xlabel("alpha")
ax.set_ylabel("raw accuracy")
ax.set_ylim(0, 1)
ax.set_title("Modulators: raw accuracy vs. alpha")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
savefig(fig, "appendix_modulators_accuracy_by_alpha.png")

# --- A.8: modulators bias-adjusted logit contrast vs alpha, E4 / E5(ratio=1) / E6 ---
fig, ax = plt.subplots(figsize=(8, 5))
for label, trials, filt in [
    ("E4 (distance)", d_e4, lambda t: True),
    ("E5 (ratio=1, i.e. no dose asymmetry)", d_e5, lambda t: t["ratio"] == 1.0),
    ("E6 (similarity)", d_e6, lambda t: True),
]:
    sub_all = [t for t in trials if filt(t)]
    alphas_m = sorted(set(_alpha_of(t) for t in sub_all))
    means = []
    for a in alphas_m:
        sub = [t["logit_contrast_adjusted"] for t in sub_all if _alpha_of(t) == a]
        means.append(np.mean(sub) if sub else float("nan"))
    ax.plot(alphas_m, means, marker="o", label=label)
ax.axhline(0, color="gray", linewidth=0.5)
ax.set_xlabel("alpha")
ax.set_ylabel("mean logit(correct) - logit(incorrect)")
ax.set_title("Modulators: bias-adjusted logit contrast vs. alpha")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)
savefig(fig, "appendix_modulators_logit_by_alpha.png")

print("\nAll figures saved to", FIGDIR)
