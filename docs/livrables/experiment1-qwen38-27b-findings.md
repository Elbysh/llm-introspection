# Experiment 1 — findings on Qwen/Qwen3.8-27B

Analysis of `results/experiment1/qwen38_all` (50,640 trials, Qwen/Qwen3.8-27B,
5 decoder blocks of 64, 4 perturbation families, 2 dose parameterizations).

Section 3.1 of the cadrage lists this model as the secondary one, to be run once the
protocol is validated on `meta-llama/Llama-3.1-8B-Instruct`. This is that run. It is
read against `experiment1-full32-findings.md`, the Llama sweep, throughout.

Provenance: blocks 3, 6, 16 from job 8162 and blocks 32, 56 from job 8163, pooled by
`code/analysis/merge_experiment1_runs.py`. Two follow-up sweeps extend it, and their
results are folded into sections 1.1 and 3 below: job 8164 (`qwen38-mid`) adds blocks
8, 10, 12; job 8165 (`qwen38-zext`) reruns blocks 3, 6, 16 on a z ladder reaching
655.36 instead of 20.48; and job 8166 (`qwen38-deep-ext`) reruns blocks 32 and 56 with
α to 1024 and z to 655.36, which is §3bis. Section 3 is written on that extended run; the original z
grid could not support the comparison it appeared to. Calibration is
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
is essentially gone by block 16 of 64, the first 25%. Relative depth is *not* the
invariant across the two models. Anyone rescaling a Llama layer choice to a deeper
model by proportion — which is what this job's first draft did — will sample mostly
dead stack.

Job 8164 added blocks 8, 10 and 12 to resolve the shape of the decay, which the first
sweep left as a jump from 0.194 at block 6 to 0.033 at block 16 with nothing between.
Mean S, **α-matched only** so that every block is compared on an identical grid:

| block | concept | random | noise | dropout | concept ÷ random |
|---|---|---|---|---|---|
| 3 | +0.3531 | +0.1828 | +0.1278 | +0.1598 | 1.9× |
| 6 | +0.2124 | +0.0822 | +0.0699 | +0.0582 | 2.6× |
| 8 | +0.1615 | +0.0693 | +0.0447 | +0.0236 | 2.3× |
| 10 | +0.1313 | +0.0432 | +0.0581 | +0.0240 | 3.0× |
| 12 | +0.0555 | +0.0554 | +0.0443 | +0.0138 | **1.0×** |
| 16 | +0.0400 | +0.0360 | +0.0291 | +0.0071 | 1.1× |
| 32 | +0.0161 | −0.0217 | −0.0131 | −0.0165 | — |
| 56 | +0.0003 | +0.0006 | −0.0001 | −0.0001 | — |

The decay is smooth rather than a cliff: concept loses roughly half its effect every
five blocks. The more interesting column is the last one. **Concept's advantage over a
fixed random direction is itself confined to the shallowest blocks and is gone by block
12**, where the two are equal to three decimal places, while both are still measurably
above zero. Whatever distinguishes a concept direction from an arbitrary one is a
property of early layers, not of the stack.

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

This is the methodological finding of the run, and the first sweep got it wrong in a
way worth recording, because the wrong version is the flattering one.

### 3.1 What the published grid showed

Pooled over all blocks at matched z, on the original ladder topping out at z = 20.48:

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

### 3.2 It was a range artefact

The control families never received enough amplitude to be detectable. Comparing the
raw amplitude the top z dose actually delivers against each family's own 75% threshold
measured on the α grid:

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

The grid overshot concept's threshold by 6 to 14 times at the live blocks and
undershot every control's by 5 to 20 times. No control was tested anywhere near its
own transition, at any block.

### 3.3 The extended sweep settles it

Job 8165 reran blocks 3, 6 and 16 with z reaching 655.36 — chosen as twice the
largest control threshold in z units, 173.6 for dropout at block 6. The controls
respond. At block 3, fixed-random:

| z | 2.56 | 10.24 | **20.48** | 40.96 | 81.92 | **163.84** | 327.68 | 655.36 |
|---|---|---|---|---|---|---|---|---|
| mean S | +0.003 | −0.006 | **−0.005** | +0.076 | +0.515 | **+0.792** | +0.644 | +0.300 |
| share + of moved | 0.64 | 0.36 | **0.42** | 0.82 | 1.000 | **1.000** | 1.000 | 0.74 |

Flat to z = 20.48, exactly reproducing the published result, and then a clean
transition immediately above where the old grid stopped. Random's z-matched peak S is
**+0.792** against concept's +0.959 — comparable magnitudes. Peak raw accuracy under
the extended z grid, every family and block:

| block | concept | random | noise | dropout |
|---|---|---|---|---|
| 3 | 0.875 | 0.967 | 0.994 | 0.875 |
| 6 | 0.984 | 0.929 | 0.969 | 0.806 |
| 16 | 0.900 | 0.933 | 0.900 | 0.850 |

**Every family reaches ceiling under z matching.** The claim that z matching isolates
conceptual content is false. The families are ordered by threshold, in z exactly as
in α.

### 3.4 What z matching actually does, quantitatively

It multiplies the family separation by the ratio of their natural scales. Since
z = α / s(l, v), a family's threshold in z is its threshold in α divided by its scale,
so at block 3 for concept against random:

```
z threshold ratio  =  (α ratio)  ×  (scale ratio)
     40.1          =    2.54     ×     15.8
                        (16/6.3)   (2.779/0.176)
```

and the measured brackets agree: concept transitions in z ∈ [1.28, 2.56], the controls
in z ∈ [40.96, 81.92], a ratio near 30. So the separation between concept and controls
is **2.5× under α matching and 40× under z matching** — and the extra factor of 16 is
arithmetic on the calibration, not a fact about the model's representations.

H1c is therefore **confirmed as stated** — the z parameterization does change the
observed differences, by a factor we can now predict exactly — while the interpretation
it invites, that the change reveals something categorical about conceptual directions,
is **refuted**. Both matchings tell the same story: every family is detectable, and
concept needs less amplitude to get there.

Recorded because the first version of this section reported §3.1 as a result. It
survived one round of checking and was caught only when the realized amplitudes were
compared against the α-grid thresholds, which is not part of the standard output.

## 3bis. At block 32 the families have opposite signs

Job 8166 reran blocks 32 and 56 with α extended to 1024 and z to 655.36, to settle
whether the deep stack was merely insensitive or genuinely inert. It was neither, and
the answer is the most interesting single result in the sweep.

**Block 56 is genuinely inert.** Raw accuracy sits at 0.487–0.519 for every family at
every amplitude up to α = 1024, roughly five times the RMS token norm there, with no
non-finite logits. Mean |S| ≤ 0.0014 with 76–80% of pairs exact ties. No attention
layer follows block 63; nothing a perturbation does at block 56 reaches the answer
token, at any amplitude. This is the cleanest null in the study.

**Block 32 is not.** Pooled over doses and both matchings:

| family | mean S | t | share positive among moved |
|---|---|---|---|
| concept | **+0.0213** | **+8.2** | 0.575 |
| noise | **−0.0366** | **−12.1** | 0.262 |
| random | **−0.0414** | **−17.9** | 0.198 |
| dropout | **−0.0443** | **−14.8** | 0.220 |

Concept is significantly **positive**; all three controls are significantly
**negative**. Both are many sigma from zero and they disagree in sign. Against dose,
α-matched, the pattern is orderly rather than noisy:

| α | 16 | 32 | **64** | 128 | 256 | 1024 |
|---|---|---|---|---|---|---|
| concept S | +0.020 | +0.027 | **+0.120** | +0.077 | −0.012 | −0.015 |
| random S | −0.028 | −0.062 | **−0.130** | −0.031 | −0.072 | −0.032 |
| random, share + of moved | 0.152 | 0.089 | **0.019** | 0.349 | 0.151 | 0.268 |

Both families reach their extremum at α = 64 and both decay above it. At that dose a
fixed random direction moves the answer **away** from the sentence it was applied to in
**98.1%** of the pairs that moved at all.

Two consequences.

First, this is why the published sweep read block 32 as "still rising": its α grid
stopped at 64, which is exactly the peak. The curve was not truncated mid-transition,
it was truncated at its maximum, and the effect never reaches the 75% criterion at any
amplitude. Concept's raw accuracy tops out at 0.616.

Second, at this depth the families differ **in the sign of the effect, not in the
amplitude needed to produce it**. At the shallow blocks every family pushes the same
way and they separate only by threshold (§2); the α-matched sign flips between block
16 (random S = +0.036) and block 32 (−0.022).

> **Corrected by §3ter.** This section originally read the sign split as the study's
> one qualitative family difference, on the reading that a generic perturbation
> degrades the targeted sentence while a concept direction adds content the answer can
> key on. The scrambled control refutes that: a coordinate permutation of a concept
> vector carries no content and still produces the positive sign. The numbers below
> stand; the interpretation is in §3ter.4.

---

## 3ter. The scrambled control: the sign is not about content

§3bis proposed that the block-32 sign split was the study's one qualitative family
difference, and §6 of the first version called it the strongest available evidence
that concept directions differ in kind. A control run says that reading is wrong.

### 3ter.1 The control

Experiment 1 varies amplitude and never content at fixed amplitude, so it cannot
separate the two on its own. A **coordinate permutation** of a concept vector can:
it preserves the L2 norm and the multiset of coordinates *exactly* while destroying
any alignment with a feature direction. Concept against scrambled at matched α is
therefore a comparison of content with magnitude held fixed by construction.

Added as the `scrambled_concept` family of Experiment 0 (opt-in, seeds recorded) and
calibrated in `results/experiment_0_calibration_qwen38_27b_scramble`, whose shared
families reproduce the original calibration bit for bit. Jobs 8172 and 8173.

### 3ter.2 Scrambling collapses the natural scale

Before a single trial, Experiment 0 reports that a permutation destroys the property
that made concept directions special in §3.4:

| block | concept | **scrambled** | fixed_random |
|---|---|---|---|
| 3 | 2.779 | **0.172** | 0.176 |
| 6 | 1.983 | **0.232** | 0.233 |
| 12 | 2.583 | **0.419** | 0.445 |
| 32 | 2.690 | **0.806** | 0.917 |

The scrambled vector lands on the fixed-random scale to within a few percent at every
block. So the ~16× scale gap driving the z artefact is **genuine alignment with the
directions along which activations actually vary**, not a consequence of how concept
vectors are built — their norm and coordinate distribution produce none of it.

### 3ter.3 Magnitude is about alignment; sign is not

Mean S, α-matched:

| block | concept | scrambled | random |
|---|---|---|---|
| 3 | **+0.4453** | +0.2545 | +0.2275 |
| 6 | **+0.2999** | +0.0716 | +0.1281 |
| 12 | +0.1122 | +0.0749 | +0.1115 |
| **32** | **+0.0257** | **+0.0218** | **−0.0270** |

Two different things happen, and they separate cleanly.

**At the shallow blocks, scrambled falls back to random.** Concept leads scrambled by
1.75× at block 3 and 4.19× at block 6, while scrambled and random are comparable.
Concept's threshold advantage is therefore about *where the direction points*, and it
is content in the only sense this design can test. This survives.

**At block 32, scrambled sides with concept.** It is positive, S = +0.0218 at
t = +7.5 with 61.9% of moved pairs positive, where random is negative at t = −16.2
with 21.7% positive. Against dose the two run together and away from random:

| α | 16 | 32 | 64 | 128 |
|---|---|---|---|---|
| concept | +0.020 | +0.027 | **+0.120** | +0.077 |
| scrambled | +0.009 | +0.015 | +0.034 | **+0.127** |
| random | −0.028 | −0.062 | **−0.130** | −0.031 |

So the sign does **not** track content. A permuted concept vector carries none, and
still produces the positive sign.

### 3ter.4 What the sign does track

The only property concept and scrambled share by construction, and a Gaussian draw
lacks, is the coordinate distribution. Measured on the bank at block 32:

| direction | kurtosis | energy in top 10 coordinates | max coordinate |
|---|---|---|---|
| concept Dust | 8.3 | 7.7% | 0.1076 |
| **scrambled Dust** | **8.3** | **7.7%** | **0.1076** |
| concept Illusions | 8.5 | 8.0% | 0.1233 |
| **scrambled Illusions** | **8.5** | **8.0%** | **0.1233** |
| fixed_random 0000 | 3.0 | 2.4% | 0.0556 |
| Gaussian reference | 3.0 | 2.2% | 0.0527 |

Concept vectors are differences of activation means, and transformer activations have
outlier dimensions, so their coordinates are heavy-tailed: kurtosis 6–8.5 against a
Gaussian's 3, with ten coordinates of 5120 carrying 6–8% of the energy against 2%. A
permutation preserves that exactly, which is why the rows are identical.

The block-32 sign therefore tracks **whether the perturbation is heavy-tailed or
Gaussian**, not whether it is meaningful. That is a property of the concept vectors'
construction, not of their semantics.

### 3ter.5 The correction

§3bis's numbers stand; its interpretation does not. The sign split is real, orderly
and many sigma from zero, and it is **not** evidence that concept directions differ
in kind. The study's one apparently categorical result has a non-semantic
explanation, and the surviving evidence for content is the quieter one: the threshold
advantage at blocks 3 and 6, which the scramble removes.

A useful follow-up, cheap and not run here: a Gaussian direction rescaled to the
concept vectors' coordinate distribution but aligned with nothing, which would test
the heavy-tail account directly rather than by elimination.

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
- The effect is confined to roughly the first quarter of the stack and decays
  smoothly, halving about every five blocks.
- **Every family is detectable under both parameterizations**, given enough amplitude.
  They differ in threshold, not in kind. The 40× separation that z matching reports is
  the 2.5× α separation multiplied by the 16× ratio of calibrated scales (§3.4).
- **Concept directions are aligned with high-variance directions, and that alignment
  is what their large calibrated scale measures.** Permuting a concept vector's
  coordinates — same norm, same coordinate multiset — collapses its scale from ~2.7 to
  ~0.17, onto the fixed-random scale, at every block (§3ter.2).
- **Concept's threshold advantage at the shallow blocks is about alignment**, and
  survives the control: concept leads its own permutation by 1.75× at block 3 and
  4.19× at block 6, while the permutation falls back to the random family (§3ter.3).
- **The block-32 sign split is not about content.** A permuted concept vector produces
  the same positive sign, and what distinguishes both from a Gaussian direction is a
  heavy-tailed coordinate distribution — kurtosis 6–8.5 against 3, ten coordinates of
  5120 carrying 6–8% of the energy against 2% (§3ter.4).
- The label-order instability that undermined the Llama headline number is absent, and
  contamination between the two sentences is an order of magnitude lower.

Corrected during the work, both recorded rather than quietly fixed: the z-matched
family comparison of §3.1, which was a grid-range artefact (§3.2–3.3), and the
interpretation of the block-32 sign in §3bis, which the scrambled control refutes
(§3ter.5). Both errors pointed the same way — toward concept directions being special
— which is the direction this project would like its results to point.

Not established:

1. **Whether the heavy-tail account of the block-32 sign is right.** It currently rests
   on elimination: content is excluded, and the coordinate distribution is the only
   remaining property concept and scrambled share. A direct test is cheap and not run
   here — a Gaussian direction reshaped to the concept coordinate distribution but
   aligned with nothing. If it reproduces the positive sign, the account holds.
2. **What the alignment at shallow blocks means.** §3ter.3 shows concept beats its own
   permutation there, so something about pointing along a real feature direction
   matters. Whether that is *the concept's* content or merely *a* high-variance
   direction is untested: the control for it is a high-variance direction carrying no
   concept, such as a leading principal component of the activations.
3. **Whether any of this is introspection.** Unchanged and outside this experiment's
   scope. The model's answer tracks which sentence was perturbed; nothing here
   distinguishes a report about an internal state from a discrimination driven by the
   perturbed tokens' downstream effects. That is what Experiments 2 and 4 are for.
