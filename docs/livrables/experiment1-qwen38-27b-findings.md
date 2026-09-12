# Experiment 1 — findings on Qwen/Qwen3.8-27B

Analysis of `results/experiment1/qwen38_all` (50,640 trials, Qwen/Qwen3.8-27B,
5 decoder blocks of 64, 4 perturbation families, 2 dose parameterizations).

Section 3.1 of the cadrage lists this model as the secondary one, to be run once the
protocol is validated on `meta-llama/Llama-3.1-8B-Instruct`. This is that run. It is
read against `experiment1-full32-findings.md`, the Llama sweep, throughout.

Provenance: blocks 3, 6, 16 from job 8162 and blocks 32, 56 from job 8163, pooled by
`code/analysis/merge_experiment1_runs.py`. Calibration is
`configs/experiment_0_calibration/qwen38_27b_full.yaml` →
`results/experiment_0_calibration_qwen38_27b`, whose manifest reports
`protocol_status: development`; these numbers are not on the frozen protocol. Unlike
the Llama calibration, this one was launched from a committed config, so Experiment 1
runs against it without `--allow_calibration_mismatch`.

Figures: `results/experiment1/qwen38_all/layer_maps/`. Tables below come from
`code/analysis/experiment1_localization_report.py`, which was checked against the
numbers already published for Llama and reproduces all of them.

---

## 0. The headline: on this model the 2AFC task is an actual task

The Llama findings open by narrowing what could be claimed. There, the model answered
"A" in 98% of perturbed trials, raw accuracy sat at 0.509 at every layer and dose, and
the entire signal lived in a sensitivity readout on the answer logits. The supportable
claim was that *the intervention shifts the A/B logit gap*, not that the model reports
anything.

That narrowing does not apply here. On Qwen, at block 3 with a concept direction at
matched α:

| α | **raw accuracy** | adjusted accuracy | tie rate |
|---|---|---|---|
| 8 | 0.884 | 0.887 | 0.087 |
| **16** | **1.000** | 0.994 | 0.013 |
| 32 | 0.981 | 0.972 | 0.019 |
| 64 | 0.934 | 0.816 | 0.069 |

Raw accuracy is the letter the model actually emits. At α = 16 it names the targeted
sentence in **every one of the 160 trials** in that cell. The 75% threshold on the raw
metric is **α = 6.26**, bracketed inside the tested grid.

The reason is visible in the sham. Qwen's standing answer bias is **+0.256 logits**
(range −0.250 to +0.625, preferring A in 85% of the 40 sham conditions) against
Llama's **+1.725** (range +0.375 to +2.750, preferring A in 100%). Llama's bias was
four times the effect being measured, which is why its argmax almost never flipped.
Qwen's is a fifth of the effect, so the decision moves.

**This is the single most important cross-model result.** The instrument that could
only be read indirectly on Llama can be read directly on Qwen.

---

## 1. H1a — displacement of the injection moves the answer

Validated, and by a wide margin. The paired estimator of doc 5.8,

```
S = (contrast when A is targeted − contrast when B is targeted) / 2
```

is what the Llama analysis established as the metric to report, because it cancels any
shift that does not depend on *which* sentence was hit. Pooled over doses and both
matchings:

| block | concept | random | noise | dropout |
|---|---|---|---|---|
| 3 | **+0.3169** | +0.0875 | +0.0618 | +0.0794 |
| 6 | **+0.1944** | +0.0395 | +0.0338 | +0.0285 |
| 16 | +0.0327 | +0.0172 | +0.0145 | +0.0034 |
| 32 | +0.0190 | −0.0156 | −0.0069 | −0.0086 |
| 56 | −0.0006 | −0.0007 | +0.0004 | +0.0001 |

All shallow-block entries are many sigma from zero (concept at block 3: t = 26.6).

**Permutation null.** Shuffling which half of each pair counts as A-targeted, holding
every logit fixed: observed mean S **+0.0554** against a null of **+0.0002 ± 0.0013**,
largest null draw 0.0037. That is **41.4 sd**; no permutation came close. The signal is
in the pairing between perturbation and target, not in the response bias.

**Block 56 is a built-in null.** Every family returns |S| ≤ 0.0007 there, and 81–88% of
its pairs are exact ties. No attention layer follows block 63, and by block 56 the
perturbation no longer reaches the answer token.

### 1.1 The live band is far shallower than Llama's, in relative depth

Llama's effect was confined to blocks 0–13 of 32 — the first 44% of the stack. Qwen's
is essentially gone by block 16 of 64, the first 25%, and concept S has already fallen
by 39% between blocks 3 and 6. Relative depth is *not* the invariant across the two
models. Anyone rescaling a Llama layer choice to a deeper model by proportion — which
is what this job's first draft did — will sample mostly dead stack.

### 1.2 The label-order control, which Llama failed, Qwen passes

On Llama, relabelling the two options — same sentences, same physical order, same
perturbation — changed measured localization by a factor of **2.4** (S = 0.379 under
AB against 0.160 under BA). That instability meant no single headline number was a
stable property of the model.

On Qwen:

| label_order | mean S | share positive among pairs that moved |
|---|---|---|
| AB | +0.0593 | 0.635 |
| BA | +0.0515 | 0.621 |

A 15% difference rather than 140%. The cosmetic relabelling is close to irrelevant
here, which is a second reason this model is the better instrument of the two.

---

## 2. H1b — at matched α the families differ, but every family gets there

Pooled over all blocks at matched α:

| family | mean S | t | share positive among moved |
|---|---|---|---|
| concept | **+0.1244** | 24.8 | 0.711 |
| random | +0.0560 | 16.0 | 0.621 |
| noise | +0.0427 | 13.9 | 0.657 |
| dropout | +0.0417 | 11.9 | 0.633 |

Concept leads by roughly a factor of two over the best control. But the ordering is a
difference in **threshold, not in kind**. At block 3 the raw-metric peak accuracies are
concept 1.000, random 0.996, noise 0.925, dropout 0.925 — every family reaches ceiling
or near it, given enough amplitude. The 75% thresholds on the adjusted metric at block
3 are concept **6.5**, dropout **32.3**, noise **63.0**, and random bracketed in
[8, 16] without a converged fit.

So the defensible statement is that a concept direction is detected at roughly a
quarter to a tenth of the amplitude a generic direction needs — not that generic
directions are undetectable. A reader who saw only the concept and random curves at
α = 16 (S = 1.306 and 0.765, raw accuracy 1.000 and 0.996) would conclude the model
detects *any* sufficiently large perturbation, and would be right.

---

## 3. H1c — the z result, and why it does not mean what it appears to

This is the methodological finding of the run, and it is a negative one.

Pooled over all blocks at matched z:

| family | mean S | t | share positive among moved |
|---|---|---|---|
| concept | **+0.1016** | 23.4 | 0.689 |
| dropout | +0.0012 | 1.5 | 0.504 |
| noise | +0.0005 | 0.8 | 0.520 |
| random | −0.0023 | −3.8 | 0.451 |

Read naively this is a spectacular confirmation of H1c: under α matching the controls
are weak but real, and under z matching they vanish to **exactly zero** while concept
is barely touched. It would say that standardizing by the natural scale isolates
conceptual content.

**It says no such thing.** The control families never receive enough amplitude to be
detectable under the z grid at all. Comparing the raw amplitude each grid actually
delivers at its top dose against each family's median calibrated scale:

| block | family | median s(l,v) | max α reached at z = 20.48 | max α on the α grid |
|---|---|---|---|---|
| 3 | concept | 2.779 | **87.6** | 64 |
| 3 | random | 0.176 | **3.7** | 64 |
| 3 | noise | 0.141 | **3.3** | 64 |
| 6 | concept | 1.983 | 74.5 | 64 |
| 6 | random | 0.233 | 5.2 | 64 |
| 16 | random | 0.501 | 10.3 | 64 |

At block 3 the entire z grid delivers at most **α = 3.7** to a fixed-random direction.
But section 2 shows random produces nothing until α = 8 and transitions at α = 16. The
z sweep stops below the amplitude at which that family would first respond. Its flat
curve is a **range artefact of the grid**, not a property of the direction.

The comparison that makes this exact: for each family, the raw amplitude the top z dose
actually delivers, against that family's own 75% threshold measured on the α grid.

| block | family | α delivered at z = 20.48 | α threshold (75%, raw) | ratio |
|---|---|---|---|---|
| 3 | concept | 87.6 | 6.3 | **14.0×** |
| 3 | random | 3.7 | bracketed in [8, 16] | **≈0.3×** |
| 3 | noise | 3.3 | bracketed in [8, 16] | **≈0.3×** |
| 3 | dropout | 1.4 | bracketed in [8, 16] | **≈0.1×** |
| 6 | concept | 74.5 | 11.6 | **6.4×** |
| 6 | noise | 5.7 | 29.4 | **0.19×** |
| 6 | dropout | 2.1 | 40.5 | **0.05×** |
| 16 | concept | 49.5 | 50.9 | 0.97× |
| 16 | random | 10.3 | 54.5 | **0.19×** |
| 16 | noise | 12.1 | 61.3 | **0.20×** |

The z grid overshoots the concept threshold by 6 to 14 times at the two live blocks and
undershoots every control's by 5 to 20 times. No control family is tested anywhere near
its own transition under z matching, at any block. The flat control curves of the table
above are that fact and nothing else.

The mechanism is the ~16× gap between concept and generic scales in this model
(2.779 against 0.176 at block 3). Dividing by s(l, v) hands concept directions sixteen
times more raw amplitude than controls at the same nominal z. The z parameterization
does not neutralize the families; on this model it systematically starves the controls.

Note the one row where the two nearly meet: concept at block 16 gets 0.97× its
threshold, and its z-matched curve there peaks at 0.747 adjusted — just short of the
75% mark, exactly as that ratio predicts. The grid's reach, not the direction's nature,
is what each curve is tracking.

**What this costs.** H1c as stated — "l'appariement en z modifie les différences
observées à α apparié" — is technically confirmed, since the numbers plainly differ.
But the interesting reading, that the difference is about the nature of the directions,
is not supported by this run. To test it the z grid must reach each control's own transition: at
block 3 that is **z ≈ 90** for random (α = 16 ÷ 0.176), and at block 16 **z ≈ 109**
(α = 54.5 ÷ 0.501) — a factor of 4.5 to 5.3 beyond the current top of 20.48.
Until then the two matchings are not comparable on this model, and the α-matched
comparison of section 2 is the only one that should carry a claim.

This is worth stating plainly because the artefact is flattering: it produces exactly
the clean, concept-is-special result the project would like to find.

---

## 4. Dose response is non-monotonic at the top

At block 3, concept, matched α, S peaks and then declines:

| α | 4 | 8 | **16** | 32 | 64 |
|---|---|---|---|---|---|
| mean S | +0.023 | +0.648 | **+1.306** | +1.111 | +0.777 |
| raw accuracy | 0.569 | 0.884 | **1.000** | 0.981 | 0.934 |

The same shape appears for random (peak at α = 16, S = 0.765, falling to 0.398 at 64).
Past the saturation point the perturbation degrades the representation faster than it
localizes, which is the "limite de fonctionnement" the cadrage anticipates for dropout
at high p, appearing here for the additive families too.

Practical consequence: a psychometric fit that assumes a monotone curve is
mis-specified above α ≈ 16 at shallow blocks. The thresholds reported in section 2 are
taken on the rising limb and are unaffected, but the top two doses should not be fed to
a monotone fit.

---

## 5. Contamination is an order of magnitude lower than on Llama

`contamination_ratio()` injects at block L on the targeted sentence and records, at
each probe block p ≥ L, the relative change of the **untargeted** sentence.

| injection | family | matching | p = L | p = 6 | p = 16 |
|---|---|---|---|---|---|
| 3 | concept | alpha | 0.000 | 0.068 | 0.152 |
| 3 | random | alpha | 0.000 | 0.053 | 0.142 |
| 3 | noise | alpha | 0.000 | 0.050 | 0.150 |
| 3 | dropout | alpha | 0.000 | 0.059 | 0.136 |

Over all 32 probe records above their injection block: median **0.067**, maximum
**0.229**. On Llama the same measurement saturated within two blocks and never decayed,
sitting at 0.5–1.0 as deep as block 30 and exceeding 1.0 for dropout.

The zero at p = L is the correctness check — the hook fires on block L's output, so the
untargeted sentence's block-L output was computed from clean inputs — and it holds.

On Qwen the two sentences are therefore much closer to independent observations, though
0.15 by block 16 is not nothing and the coverage is as thin as it was on Llama (one
record per layer/family/matching, top dose only, single pair and presentation).

---

## 6. What this run does and does not establish

Established:

- The 2AFC localization task works **behaviourally** on this model, reaching raw
  accuracy 1.000 at block 3. H1a is confirmed at 41 sd against a permutation null.
- The effect is confined to roughly the first quarter of the stack, shallower in
  relative depth than Llama's first 44%.
- At matched α the families are ordered concept > random > noise ≈ dropout, as a
  difference in detection threshold — every family reaches ceiling eventually.
- The label-order instability that undermined the Llama headline number is absent.
- Contamination between the two sentences is an order of magnitude lower than Llama's.

Not established, and needing more runs:

1. **The z-matched comparison.** Section 3: the grid starves the control families. Rerun
   with z extending to ~90 before any family comparison at matched z is quoted.
2. **Blocks between 6 and 16.** S falls from 0.194 to 0.033 across that gap with nothing
   sampled in between, so the shape of the decay is unknown. Blocks 8, 10, 12 would fix
   it at about one hour each.
3. **Block 32's late onset.** S there is +0.019 pooled but +0.120 at the top α dose
   alone, still rising. The deep stack is not inert, merely far less sensitive; the α
   grid would need to extend past 64 to characterize it.
4. **Whether concept directions are special at all.** Section 2 shows a threshold
   advantage, not a categorical difference, and section 3 removes the evidence that
   looked categorical. This is the open question the project actually cares about, and
   this run narrows it rather than settling it.
