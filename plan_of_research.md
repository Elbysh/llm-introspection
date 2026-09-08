# Multi-Injection / Multi-Steering: Experimental Plan

## 0. Context

This plan investigates whether a language model can introspect on **multiple simultaneous concept injections** made at different layers — extending single-concept injection work (Lindsey, *Emergent Introspective Awareness in LLMs*, Anthropic, Oct. 2025; Pearson-Vogel et al., *Latent Introspection: Models Can Detect Prior Concept Injections*, Feb. 2026), both of which only inject one concept at a time. Throughout the experiments, we will work on the same model : Meta-Llama-3.1-8B-Instruct.

**Injection method:** for each injection *k*, at layer *l*, with unit-normalized steering vector *v_k*:

```
h(l) <- h(l) + alpha_k * v_k
```

**Note on "ordering" (H3):** "concept A arrives before concept B" refers strictly to **layer depth** (A injected at a shallower layer than B, i.e. processed earlier in the network's depth), not to token position. Token position is held fixed and out of scope for this plan.

---

## 1. Known methodological risks (apply across all blocks)

- **Self-report ≠ ground truth.** Concept injection can perturb the model's entire output distribution (raised entropy, generic "yes"-bias), so a positive self-report is not by itself evidence of concept-specific detection. Every detection/identification experiment needs a **sham condition** (random unit vector, no semantic content, same alpha) to separate "detects a generic anomaly" from "detects this specific concept."
- **Naive free-response elicitation is weak.** Directly asking "do you notice an injection?" tends to default to "no" even when the information is present and decodable from the residual stream. A **calibration block** (Block 0) is needed before drawing conclusions from null results in later blocks.
- **Self-report should be cross-checked against a non-verbal readout.** Where feasible, run a linear probe or logit-lens decode (presence / identity / relative depth) on the same activations, in parallel with the verbal self-report, to distinguish "no information present" from "information present but not reported."
- **Open-ended grading needs an objective, blind criterion** (e.g., embedding similarity between the free-response answer and a canonical concept description, pre-registered threshold, or a separate blind LLM judge).
- **Statistical power.** Use multiple concepts (ideally spanning varied semantic categories) and multiple layer pairs, with randomized concept↔layer assignment across trials, not a single fixed pair.

---

## Block 0 — Calibration (prerequisite)

Single-concept injection replication (à la Lindsey) to find, for this model, the layer(s) and alpha range where self-report is reliably above the false-positive rate. Provides the baseline operating point for all later blocks and guards against drawing conclusions from a poorly elicited null result.

- Sweep: single injection, layer ∈ {full range}, alpha ∈ {low → high}.
- Prompt: `"Describe your current internal state. Do you notice anything unusual — any thought, concept, or idea that feels like it doesn't belong, that you didn't intend to think about? If so, what is it?"`
- Output: detection accuracy and false-positive rate as a function of (layer, alpha) → select operating window for Blocks 1–4.

---

## Block 1 — Detection (H1)

**H1:** The model can detect the presence of multiple injections made at different layers, independently of their type and layer position.

### E1 — Count report
K (varied) injections at different layers, equal alpha.

**Prompt:**
`"I may or may not have injected one or more concept vectors into your internal activations while processing this message. Do you detect any injected content? If so, how many distinct injections do you notice?"`

**Controls:**
- **C1.1 — Null injection (k=0):** same prompt, no injection → false-positive rate.
- **C1.2 — Reported vs. actual k:** run k ∈ {0,1,2,3,4}, build a confusion matrix of reported vs. actual count.
- **C1.3 — Rephrasing check:** reformulate to avoid yes/no demand characteristics, e.g. `"On a scale from 0 to 10, how much does your current processing feel different from normal?"` — checks that results aren't an artifact of a binary "yes"-biased phrasing.
- **C1.4 — Sham injection:** replace the k concept vectors with k random unit vectors of the same alpha, no semantic content → isolates generic-anomaly detection from concept-specific detection.

---

## Block 2 — Identification (H2)

**H2a (weak):** the model finds the injected concepts to be different from each other.
**H2b (strong):** the model names and/or defines the concepts.

### E2 — Free identification
2 injections at different layers, equal alpha.

**Prompt:**
`"I have injected exactly two concept vectors into your internal activations. What are they? Name or describe each one as precisely as you can."`

**Controls:**
- **C2.1 — Single concept:** same prompt adapted to "exactly one" → upper bound on per-concept detectability, ties back to Block 0.
- **C2.2 — Forced choice:** `"I have injected two concepts into your activations. Here is a list of candidates: [A, B, C, D, E, F, G, H, I]. Which two were injected?"` (numbered list, order randomized per trial) — scaffolded format shown to substantially raise sensitivity over free response.
- **C2.3 — Sham identification:** one real concept + one sham (random) injection → tests whether the model falsely "identifies" content in the sham slot.

**Grading:**
- Free response (E2, C2.1): embedding similarity between the answer and a canonical concept description, pre-registered threshold; ideally scored by a blind judge unaware of the injected concept.
- Forced choice (C2.2): exact-match accuracy against ground truth.

---

## Block 3 — Relative Layer-Depth Ordering (H3)

**H3:** The model can report the relative layer-depth ordering of two injections (A injected at a shallower layer than B).

### E3 — Two-alternative forced choice
Same setup as E2 (2 injections, different layers, equal alpha).

**Prompt:**
`"I made two injections into your activations, one at an earlier processing stage (shallower layer) and one at a later one (deeper layer). Between {concept_1} and {concept_2}, which one entered your processing first?"`

**Controls:**
- **C3.1 — Presentation order permutation:** swap the order in which concept_1/concept_2 are named in the prompt, to detect a recency/primacy bias in the response independent of any real signal.
- **C3.2 — Chance baseline:** binomial test against 50% accuracy; ensure enough trials for adequate power.

---

## Block 4 — Modulators (H4, H5, H6)

Factorial extensions of Block 2 (identification) and Block 3 (ordering).

### E4 — Effect of layer distance |i−j| (H4)
**H4:** Detection/ordering ability is strongly dependent on the layer distance between injections.

Repeat E2 and E3 while systematically varying |i−j|. Plot accuracy vs. |i−j| for both tasks.

**Controls:**
- **C4.1 — Fixed distance, varying absolute position:** test a constant |i−j| both early and late in the network, to confirm the effect tracks *distance* rather than *absolute depth*.
- **C4.2 — Extreme distances:** |i−j| = 1 and |i−j| = L−1.

### E5 — Effect of relative injection strength (H5)
**H5:** Detection/ordering ability depends on the relative strength of each injection (alpha_A vs. alpha_B).

Repeat E2 at fixed layers, varying the ratio alpha_A / alpha_B.

### E6 — Effect of concept similarity (H6, bonus)
**H6:** Detection ability may depend on the semantic proximity of the injected concepts.

Repeat E2 with concept pairs selected across a range of cosine similarities between their steering vectors. Plot accuracy vs. cosine similarity. Include a near-identical concept pair as a limit case (does the model merge them into a single perceived concept?).

---

## 5. Cross-cutting validity checks (apply to every block)

1. **Sham/null conditions** wherever a real injection is used (see C1.4, C2.3), to separate concept-specific signal from generic perturbation.
2. **Probe / logit-lens cross-validation** on a subsample of trials: decode presence, identity, and relative depth directly from activations, and correlate this "objective" signal with the verbal self-report — the only way to tell whether a self-report failure reflects a genuine introspective limit or an elicitation problem.
3. **Blind grading** for all free-response scoring (concept identification, ordering justification if collected).
4. **Randomized concept↔layer assignment** across trials to avoid confounding a specific concept with a specific layer.

---

## 6. Summary Table

| Hypothesis | Block | Main experiment | Key controls |
|---|---|---|---|
| — | 0 | Single-injection calibration | Layer × alpha sweep |
| H1 | 1 | E1 — count report | C1.1 null, C1.2 confusion matrix, C1.3 rephrasing, C1.4 sham |
| H2a/H2b | 2 | E2 — free / forced-choice identification | C2.1 single concept, C2.2 candidate list, C2.3 sham slot |
| H3 | 3 | E3 — 2AFC layer-depth ordering | C3.1 presentation order, C3.2 chance baseline |
| H4 | 4 | E4 — accuracy vs. \|i−j\| | C4.1 fixed distance/position, C4.2 extreme distances |
| H5 | 4 | E5 — accuracy vs. alpha_A/alpha_B | — |
| H6 | 4 | E6 — accuracy vs. cosine similarity | Near-identical concept pair |

