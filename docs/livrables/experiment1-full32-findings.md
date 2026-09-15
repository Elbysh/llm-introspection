# Experiment 1 — findings from the `full32_all` sweep

Analysis of `results/experiment1/full32_all` (306,280 trials, Llama-3.1-8B-Instruct,
32 decoder blocks, 4 perturbation families, 2 dose parameterizations).

Provenance, from the run's own note: blocks 0–30 come from job 8129, block 31 from
job 8130, and the concept family is absent at block 31 because its `*_32_*.pt` does
not reproduce the norm Experiment 0 recorded. Calibration is
`configs/experiment_0_calibration/development_full.yaml`, whose manifest reports
`protocol_status: development` — these numbers are not on the frozen protocol.

Figures: `results/experiment1/full32_all/layer_maps/`, produced by
`code/analysis/plot_experiment1_layer_maps.py`.

---

## 0. What the reported metric actually measures

This governs how every number below can be phrased, so it comes first.

The model answers **"A" in 98% of perturbed trials**. Raw 2AFC accuracy is **0.509**
overall and stays flat at every layer and every dose. Nothing in this run shows the
model *reporting* where it was perturbed.

The entire signal lives in `accuracy_adjusted`, which scores the sign of
`(logit_A − logit_B) − sham`. That is a **sensitivity readout on the answer logits**,
not a behavioural response. The supportable claim is:

> the intervention shifts the A/B logit gap toward the targeted sentence

and not

> the model can detect or report which sentence was perturbed.

Section 1 shows that even the first claim needs a better estimator than
`accuracy_adjusted`.

---

## 1. The letter-A bias, and what happens when A and B swap roles

### 1.1 The bias is in the letter, not the position

All 20 distinct sham conditions have `logit_A > logit_B`, with a margin of 0.375 to
2.75 and a mean of **1.725 logits**. The bias holds in *both* label orders — and since
`label_order = "BA"` prints the first sentence labelled **B**, a preference that
survives the swap is a preference for the **letter A**, not for the first position.
The `label_order` control did its job and caught this.

### 1.2 The sham subtraction does not remove it

`accuracy_adjusted` subtracts the sham contrast per condition, which should neutralise
a constant bias. It does not. Pooled over the live band (blocks 0–13):

| targeted sentence is labelled | accuracy_adjusted | n | vs chance |
|---|---|---|---|
| **A** | **0.481** | 67,760 | −10.0σ (*below* chance) |
| **B** | **0.753** | 67,760 | +152.9σ |

Split by *physical* position instead, the two conditions are indistinguishable:
target_index 0 → 0.6178, target_index 1 → 0.6164. So the 0.27 gap is entirely about
which **letter** labelled the targeted sentence.

### 1.3 Why: the perturbation has a generic direction

Mean adjusted contrast in the live band is **−0.310** — perturbing *anything* tends to
pull the logit gap *down*, away from the default answer A. Broken out:

| targeted | mean adjusted contrast |
|---|---|
| A | −0.041 |
| B | −0.580 |

A generic "disruption reduces confidence in the default answer" effect scores as a
**hit** on every B-targeted trial and a **miss** on every A-targeted trial. Averaged
over a balanced design that lands near 0.62 and looks like localization. Roughly half
of what `accuracy_adjusted` reports in the live band is this artefact.

### 1.4 The fix is already in `summary.json`

`summarize()` computes the paired statistic of doc 5.8:

```
S = (contrast when A is targeted − contrast when B is targeted) / 2
```

over the two trials that share everything but the targeted sentence. Any shift that
does not depend on *which* sentence was hit cancels exactly. It is stored as
`mean_localization_contrast` — and it is **not** what `accuracy_adjusted` or the
current figures show.

Under the paired estimator the effect is real and the family ordering survives, but
the controls are much weaker than `accuracy_adjusted` suggested (live band, blocks 0–13):

| family | mean S | t | share S > 0 | accuracy_adjusted said |
|---|---|---|---|---|
| concept | **0.388** | 122.1 | **0.723** | 0.686 |
| noise | 0.222 | 64.1 | 0.575 | 0.595 |
| dropout | 0.218 | 60.4 | 0.550 | 0.586 |
| random | 0.178 | 62.6 | **0.528** | 0.561 |

On the same slice, the concept-versus-random gap is *larger* under the correct
estimator (+0.195 on share-positive, against +0.125 on `accuracy_adjusted`), because
the artefact inflated the controls more than it inflated concept. **Report `mean_localization_contrast`, not
`accuracy_adjusted`.**

### 1.5 Swapping the letters still halves the signal

Even after pairing, which letter is printed first matters:

| label_order | mean S | share S > 0 | n |
|---|---|---|---|
| AB (first sentence is "A") | **0.379** | 0.657 | 33,880 |
| BA (first sentence is "B") | **0.160** | 0.565 | 33,880 |

while physical presentation order is perfectly balanced (S = 0.271 vs 0.268).

So a cosmetic relabelling of the two options — same sentences, same physical order,
same perturbation — changes the measured localization by a factor of **2.4**. The
effect is robustly positive in both, but its magnitude is not a stable property of the
model. Any single headline number should be reported per label order, or the design
should be extended until the swing is understood.

### 1.6 Is any of this just the bias? Two null controls

The obvious objection: the model wins with A 98.7% of the time, so perhaps the whole
effect is the bias being re-described. Two controls say otherwise.

**Permutation test.** Shuffle which sentence is labelled "targeted", holding every logit,
the bias, and the pipeline fixed. If the effect came from the bias it would survive,
since only the target assignment changes:

| | mean S (concept, live band) |
|---|---|
| observed | **+0.3877** |
| shuffled null (200 draws) | −0.0003 ± 0.0064 |

The observed value is 60 sd above the null and no permutation reached it. The signal is
in the pairing between perturbation and target, not in the bias.

**The dead band is a built-in null.** The same pipeline on blocks 14–30 returns
S = 0.000–0.003, and identically zero at block 31. A metric that manufactured signal
from a biased model would show it at every depth.

**The argmax does flip, rarely.** B wins outright in 4,038 live-band trials (3.0%), and
**75.9% of those flips are on B-targeted trials** (+32.9σ). In the dead band the flips
nearly vanish (56 trials). So there is behavioural evidence, confined to the tail where
the perturbation overcomes the bias.

**The reconciling magnitude.** The sham A/B margin is 1.725 logits; the concept
perturbation moves the gap by 0.4–0.53, roughly a quarter of it — enough to be measured
at 122σ, not enough to flip a decision. That is why raw accuracy stays pinned at 0.5
while S is large. The two facts are consistent.

The supportable claim is therefore narrow but not empty: *the perturbation systematically
moves the answer logits toward the targeted sentence, by about a quarter of the model's
standing answer bias, in blocks 0–13 only.*

---

## 2. Contamination: the two sentences are not independent observations

### 2.1 What was measured

`contamination_ratio()` injects at layer L on the targeted sentence and, at every probe
block p ≥ L, records

```
ratio(p) = || activations of the UNTARGETED sentence, perturbed − clean || / || clean ||
```

A value of 1.0 means the untargeted sentence's representation moved by as much as its
own norm.

Coverage is thin. The budget produces one record per (layer, family, matching), which
with 4 families × 2 matchings exhausts 24 records after **3 injection layers (0, 1, 2)**,
at the largest dose only, and only for `pair_id 0`, `order 0`, `label_order AB`,
`target_index 0`. There is no contamination data for blocks 3–31.

### 2.2 The result: contamination is essentially total

| injection | family | matching | p=L | p=L+2 | p=8 | p=20 | p=30 |
|---|---|---|---|---|---|---|---|
| 0 | concept | alpha | 0.000 | 0.961 | 1.061 | 0.843 | 0.973 |
| 0 | dropout | alpha | 0.000 | 1.470 | 0.922 | 0.653 | 0.673 |
| 0 | random | alpha | 0.000 | 1.121 | 1.033 | 0.778 | 0.871 |
| 1 | random | alpha | 0.000 | 1.321 | 1.111 | 0.988 | 1.023 |
| 0 | noise | alpha | 0.000 | 0.648 | 0.695 | 0.568 | 0.646 |
| 2 | concept | alpha | 0.000 | 0.682 | 0.892 | 0.634 | 0.824 |

Three things to read off it:

1. **The zero at p = L is a correctness check, not a result.** The hook fires on block
   L's output, so the untargeted sentence's block-L output was computed from clean
   block-(L−1) inputs. It must be 0.000, and it is.

2. **Contamination saturates within two blocks** and then never decays — it is still
   0.5–1.0 at block 30. Several cells exceed **1.0** (dropout at L=0 reaches **1.470**),
   meaning the untargeted sentence carries more perturbation than clean signal.

3. At these doses the 2AFC question is **ill-posed**. "Which sentence was directly
   targeted?" presumes the other one is a control. At α=128 it is not a control; it has
   been overwritten too. The localization that survives (section 1.4) is the model
   tracking *which sentence was hit first and hardest*, not a clean contrast between a
   perturbed and an unperturbed item.

### 2.3 Contamination is structurally one-directional

Only `target_index = 0` is probed — perturb the first sentence, measure the second.
The reverse is **exactly zero by construction**: attention is causal, so tokens of the
first sentence never attend to tokens of the second. Perturbing the second sentence
cannot contaminate the first.

The two target conditions are therefore not symmetric, and no amount of trial balancing
fixes it. It is worth stating explicitly that this is a *positional* asymmetry —
and section 1.2 showed that physical position has **no** measurable effect on accuracy
(0.6178 vs 0.6164), while the *letter* has a huge one. The two asymmetries are separate
phenomena, and the one that actually moves the numbers is the letter, not causality.

### 2.4 A scaling artefact visible in the same table

At injection layer 0 under z-matching, dropout / noise / random contaminate by only
0.05–0.19, while concept contaminates by 0.79. The reference scale `s(layer 0)` is tiny
for the fixed-random bank, so the same z buys a far smaller absolute perturbation. At
block 0, "matched z" is not matching the families to each other in any physical sense.

---

## 3. Headline results

### 3.1 A hard depth gate at block 14

Blocks 0–13 localize; blocks 14–30 are at chance (0.4993 pooled over 164,560 trials);
block 31 is identically zero. `mean_localization_contrast` makes it starker than
accuracy does — concept runs 0.35–0.57 through blocks 1–13 and then drops to **±0.01**
from block 14 up.

This is not a bfloat16 quantization floor. Logits are quantized to ~0.0625 and the tie
rate climbs from 0.17 (blocks 0–13) to 0.60 (14–30) to 1.000 (31). But **excluding ties
entirely**, blocks 14–30 sit at 0.495–0.506 — within 1.6σ of chance for every family,
on ~30,000 resolvable concept trials. The dead band is real.

### 3.2 It is not the perturbation washing out

Mean |adjusted contrast| at the top dose decays *smoothly* with depth: 1.60 at block 0,
0.86 at block 13, 0.52 at block 14, 0.06 at block 30. Accuracy goes 0.77 → 0.50 across
that same single block. The perturbation still lands at block 14 with half the magnitude
it had at block 13; what disappears is the *information about which sentence*, not the
magnitude. See `effect_*.png`.

### 3.3 Block 31 is structurally dead, not broken

Tie rate is exactly 1.000 at every dose, including realized amplitude 127. The hook sits
on the decoder block output (`experiment1_psychometrics.py:319`), so block 31's output at
sentence positions is followed only by the final norm and `lm_head`, both position-wise.
It can never reach the answer token. The missing concept cell at block 31 costs nothing.

---

## 4. Anomalies

### 4.1 The "concept" family is one signed axis, and one concept is its negative pole

Per-concept, matched z, blocks 0–13:

| Dust | Satellites | recursion | shutdown |
|---|---|---|---|
| **0.540** | 0.827 | 0.853 | 0.892 |

Dust is inert at matched **alpha** too (0.535, at an identical 49.6 realized amplitude),
so this is not a scale artefact — it is the direction itself.

Loading the saved vectors explains it. Across blocks 2–14, PC1 captures **63–86%** of
the variance among the four concept vectors, and **Dust always loads with the opposite
sign** to the other three:

```
block 11:  Dust·recursion = -0.75    recursion·shutdown = +0.75
           Dust·shutdown  = -0.64    Satellites·recursion = +0.26
           Dust·Satellites= -0.32
```

These are not four independent concept probes; in the live band they are approximately
one axis with three positive samples and one negative. Note that Satellites has the
*weakest* loading magnitude yet still scores 0.83 — so it is the **sign** that predicts
success, not the magnitude.

The concept-versus-control gap should therefore be read as *"one privileged signed
direction localizes in blocks 9–13"*, not as *"the model introspects concepts."*

**Decisive and cheap follow-up: inject −Dust.** If it jumps to ~0.85, the effect is a
signed axis and the concept labels are incidental. The sweep used only positive doses,
so this run cannot distinguish the two hypotheses.

### 4.2 Fixed random directions systematically *anti*-localize at blocks 7 and 12

`fixed_random__block_07__0000` scores **0.05** at α = 1–2 — 95% of trials move the
logits toward the *wrong* sentence, at **−16.9σ**.

All **13** significantly-below-chance cells in the entire 2,794-cell sweep belong to the
`random` family, all at blocks 7 and 12, at mid doses, in both matchings. Concept, noise
and dropout never invert; and at blocks 7 and 12 all *three* random directions invert,
so it is a property of the depth, not of one unlucky vector.

Plausible mechanism: a *coherent* push (the same vector on every token) at those depths
degrades the targeted sentence enough that the model names the one it can still read.
Renewed noise is incoherent across tokens and does not do it; concept directions are
coherent but do not either. Unexplained, and worth a dedicated probe.

### 4.3 The psychometric model is the wrong shape

At matched alpha, **all 14 live layers** for random / noise / dropout peak *before* the
largest dose and then fall back, losing 0.16–0.21 accuracy. The curves are
inverted-U, but `fit_psychometric` fits a monotone logistic. That is why
`threshold_reportable` is 0–16% across the sweep: the fits are not failing on noise,
they are failing on shape. No 75% threshold from this run should be quoted.

Concept is markedly more monotone (mean drop-off 0.049 at alpha, 0.022 at z, against
0.16–0.21 for the controls) — another concept-specific signature: concept perturbation
*saturates*, the controls actively *reverse*.

### 4.4 The dropout dose axis is degenerate at the top

`dropout_rate` reaches **0.9971** at α=128 (max 0.9999 across layers). The top three or
four doses are all "sentence obliterated"; p cannot exceed 1, so they are not dose
increases. Dropout's fall from 0.81 at α=16 to 0.52 at α=128 is that ceiling, not a
dose–response. Realized amplitude tracks the request up to α=32 (31.1 vs 32) and then
falls short (109.7 vs 128) for the same reason.

### 4.5 The sham rows are duplicated

`trials.csv` holds 40 sham rows but only **20 distinct conditions**, each appearing twice
with byte-identical logits — the two-job stitch wrote a sham block per job. Harmless for
the adjusted metric (the forward pass is deterministic, so the same value is subtracted),
but `summary.json` reports `sham.n_trials: 40` when there are 20 measurements. Fix before
anyone quotes it.

### 4.6 Block 16 sits below chance

Excluding ties, block 16 is at 0.466 (**−5.0σ**) and block 14 at 0.479 (−3.1σ), while
block 17 is at 0.529 (+4.0σ). Over 17 tests block 16 survives a multiple-comparisons
correction. Small in absolute terms, but not noise.

### 4.7 Unequal power across families

Cells are not equally powered: concept has 160 trials/cell (4 directions), random 120
(3), noise and dropout 80 (2). Worth carrying into any family comparison.

---

## 5. What to change

1. **Make `mean_localization_contrast` the reported metric.** It is already computed and
   is immune to the letter bias that distorts `accuracy_adjusted` (§1.4).
2. **Report per label order, or explain the 2.4× AB/BA swing** (§1.5).
3. **Run the −Dust control.** One condition, and it decides whether §4.1 is about
   concepts or about a signed axis.
4. **Raise `--contamination_trials`** so the diagnostic covers the live band (blocks
   3–13) and more than one dose, and document that contamination at the top doses makes
   the 2AFC question ill-posed (§2.2).
5. **Drop the 75% threshold reporting**, or replace the monotone logistic with a model
   that admits a non-monotone dose response (§4.3).
6. **Cap the dropout dose grid** where the rate saturates (§4.4).
7. **Fix `sham.n_trials`** on the stitched summary (§4.5).
