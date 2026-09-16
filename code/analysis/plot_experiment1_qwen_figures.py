"""Qwen3.8-27B figures for report.tex, on the paired estimator S (doc 5.8)."""
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/raid/home/students/gegout_tho/sujet_15/llm-introspection")
FIG = ROOT / "docs/livrables/figures/experiment1-qwen"
FIG.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 160, "savefig.bbox": "tight"})
LAB = {"concept": "concept", "random": "aléatoire fixe", "noise": "bruit",
       "dropout": "dropout", "scrambled": "concept permuté"}
COL = {"concept": "#1b6ca8", "random": "#c8553d", "noise": "#3c896d",
       "dropout": "#8a6bbe", "scrambled": "#d98b28"}
KEY = ["layer", "family", "matching", "dose", "direction_id", "pair_id", "order", "label_order"]


def paired(run):
    t = pd.read_csv(ROOT / f"results/experiment1/{run}/trials.csv", low_memory=False)
    p = t[t.kind == "perturbed"].copy()
    p["signed"] = np.where(p.target_label == "A", p.contrast, -p.contrast)
    g = p.groupby(KEY, sort=False).agg(s=("signed", "sum"), n=("signed", "size")).reset_index()
    g = g[g.n == 2].copy(); g["S"] = g.s / 2.0
    g["layer"] = g.layer.astype(int)
    return g, p


# ---- Fig 1: depth profile, alpha-matched (identical grid at every block) ----------
a, _ = paired("qwen38_all"); b, _ = paired("qwen38-mid")
dep = pd.concat([a, b])
dep = dep[dep.matching == "alpha"]
fig, ax = plt.subplots(figsize=(5.0, 2.9))
for f in ["concept", "random", "noise", "dropout"]:
    d = dep[dep.family == f].groupby("layer").S.mean()
    ax.plot(d.index, d.values, "o-", ms=4, lw=1.4, color=COL[f], label=LAB[f])
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("bloc décodeur $\\ell$ (sur 64)")
ax.set_ylabel("$S$ moyen (logits)")
ax.set_title("Profil de profondeur, $\\alpha$ apparié", fontsize=9)
ax.legend(fontsize=7)
ax.axvspan(20, 60, color="0.88", zorder=0)
ax.text(40, 0.07, "pile morte", ha="center", fontsize=8, color="0.45")
fig.savefig(FIG / "depth_profile.pdf"); plt.close(fig)

# ---- Fig 2: the z range artefact -------------------------------------------------
# the extended ladder alone: one uniform grid, 160 trials per dose, no run pooling
z, _ = paired("qwen38-zext")
fig, ax = plt.subplots(figsize=(5.0, 2.9))
sub = z[z.layer == 3]
for f in ["concept", "random", "noise", "dropout"]:
    d = sub[sub.family == f].groupby("dose").S.mean()
    ax.plot(d.index, d.values, "o-", ms=4, lw=1.4, color=COL[f], label=LAB[f])
ax.set_xscale("log", base=2)
ax.axhline(0, color="k", lw=0.6)
ax.axvline(20.48, color="#b03a2e", ls="--", lw=1.2)
ax.text(20.48, 0.93, " ancienne\n grille s'arrête ici", transform=ax.get_xaxis_transform(),
        fontsize=7, color="#b03a2e", va="top", ha="left")
ax.set_xlabel("$z$ (bloc 3)")
ax.set_ylabel("$S$ moyen (logits)")
ax.set_title("Artefact d'étendue de grille en $z$", fontsize=9)
ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(0.01, 0.62),
          framealpha=0.95)
fig.savefig(FIG / "z_range_artefact.pdf"); plt.close(fig)

# ---- Fig 3: block-32 sign split and the scrambled control -------------------------
sd, _ = paired("scram-deep"); ss, _ = paired("scram-shallow")
de, _ = paired("qwen38-deep-ext")
fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.9))
# left: S vs alpha at block 32, concept / scrambled / random
sub = pd.concat([sd, de])
sub = sub[(sub.layer == 32) & (sub.matching == "alpha")]
for f in ["concept", "scrambled", "random"]:
    d = sub[sub.family == f].groupby("dose").S.mean()
    if d.empty: continue
    ax[0].plot(d.index, d.values, "o-", ms=4, lw=1.4, color=COL[f], label=LAB[f])
ax[0].set_xscale("log", base=2); ax[0].axhline(0, color="k", lw=0.8)
ax[0].set_xlabel("$\\alpha$"); ax[0].set_ylabel("$S$ moyen (logits)")
ax[0].set_title("Bloc 32 : le signe sépare les familles", fontsize=9)
ax[0].legend(fontsize=7)
# right: scrambled falls back to random when shallow, sides with concept when deep
rows = []
for blk, src in [(3, ss), (6, ss), (12, sd), (32, sd)]:
    s2 = src[(src.layer == blk) & (src.matching == "alpha")]
    for f in ["concept", "scrambled", "random"]:
        v = s2[s2.family == f].S
        if len(v): rows.append({"layer": blk, "family": f, "S": v.mean()})
r = pd.DataFrame(rows)
x = np.arange(4)
for k, f in enumerate(["concept", "scrambled", "random"]):
    d = r[r.family == f].set_index("layer").reindex([3, 6, 12, 32]).S
    ax[1].bar(x + (k - 1) * 0.27, d.values, 0.26, color=COL[f], label=LAB[f])
ax[1].set_xticks(x); ax[1].set_xticklabels(["bloc 3", "bloc 6", "bloc 12", "bloc 32"], fontsize=8)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set_ylabel("$S$ moyen (logits)")
ax[1].set_title("Le contrôle par permutation", fontsize=9)
ax[1].legend(fontsize=7)
fig.savefig(FIG / "sign_split_scrambled.pdf"); plt.close(fig)
print("wrote:", sorted(p.name for p in FIG.glob("*.pdf")))
