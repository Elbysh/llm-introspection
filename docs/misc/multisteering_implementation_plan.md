# Multi-Steering: Adapting the Codebase to Experiments 9-11

## Context

`plan_of_research.md` was rewritten wholesale into a psychophysics protocol (Experiences
0-11). Multi-steering is no longer "Blocks 1-4" — it is now:

- **Experience 9** (`plan_of_research.md` §14) — counting distinct injections, k in {0..4}.
- **Experience 10** (§15) — identifying the content of two injected concepts.
- **Experience 11** (§16) — reporting the relative layer-depth order of two injections,
  plus modulators (layer distance, absolute position, dose ratio, concept-pair similarity).

This document replaces the previous version of this plan (which targeted the old
"Block 0-4 / H1-H6" structure) and replaces `next_steps.md`, whose to-do list was written
against that old structure and against a calibration gate this pass no longer uses.

**Scope decision: z ≡ alpha, no natural-scale calibration.** The new plan's dosing formula
is `alpha = z * s(l, v)`, where `s(l, v)` is the natural variability of un-injected
activations (Experience 0, §4). We are **not** implementing Experience 0 for this pass. `z`
plays exactly the role the old code's `alpha` already plays: a raw amplitude multiplying a
unit-normalized steering vector. Every place the plan writes `z`, this codebase reads
`alpha`. CLI flags stay named `--alpha` (not `--z`) to avoid a pure-rename diff across four
scripts and their sbatch jobs — this document just makes the equivalence explicit. One
consequence: `calibration.py` / `jobs/calibration.sbatch` (the old Block 0 self-report
sweep) is **not** a prerequisite gate for Exp 9-11 anymore, since there is no scale-derived
operating window to compute from it. It can still be eyeballed informally to pick a
reasonable alpha range, but nothing here depends on rerunning it.

Everything below is scoped to Exp 9-11 only. Experiences 0-8 (single-injection
psychophysics — 2AFC localization, presence detection, alpha vs. z comparison, the pilot in
§17.6) are a separate workstream and out of scope here.

## What's already reusable, unchanged

From the Phase 1 infra (already implemented, still correct for this pass):

- `code/utils/multi_inject.py` — the K-way injection engine (`InjectionSpec`,
  `apply_multi_injection`). No changes needed.
- `code/utils/sham_vectors.py` — `make_random_direction`, `infer_hidden_dim`. No changes.
- `code/utils/embedding_judge.py`, `code/utils/concepts.py` — concept registry, cosine
  similarity, distractor/bucket helpers. No changes.
- `code/utils/response_parsing.py`'s `is_coherent`, `parse_count` — kept as-is (see Exp 9
  below for why `parse_count` doesn't need touching).

## Per-experiment deltas

### Experience 9 — counting (`code/experiments/multi_detection.py`)

Current script implements the old Block 1 (`E1` count-report + `C1.1-C1.4`). Required
changes:

1. **Prompt.** `all_prompts.py:get_count_report_messages()` currently asks an open "do you
   detect any injected content? If so, how many..." question. Replace with §14.3's exact
   wording, which forces a single-token numeric answer ("Answer with exactly one number: 0,
   1, 2, 3, or 4"). `response_parsing.parse_count` already extracts a bare digit or number
   word 0-4 and needs no change — it just gets a cleaner signal once the prompt is tightened.
2. **Two dosing regimes (§14.2).** Add `--dose_regime {individual,budget}`:
   - `individual` (current behavior): every injection gets the same `alpha` (=z).
   - `budget` (new): fix `--z_total`, and for `k>0` set each injection's
     `alpha_i = z_total / sqrt(k)`; `k=0` is unaffected (no formula applied, matching
     §14.2's note that the sham has no injections to scale).
   `build_specs()` needs to branch on this and `run_detection()` needs to record
   `dose_regime`, `z_total` (when applicable), and the per-injection `alpha_i` list rather
   than a single scalar `alpha` (today it records one shared `alpha` for all k slots, which
   is only correct for the `individual` regime).
3. **Layer pool.** `DEFAULT_LAYER_POOL = list(range(0, 32, 2))` only covers 16 of the 32
   layers. §14.2/3.6 require drawing from all 32 blocks. Change the default to
   `list(range(0, 32))`; `--layer_pool` can still be narrowed via CLI for cheaper runs.
4. **Terminology fix, not a behavior change.** The script's `condition="sham"` currently
   means "k random directions instead of k concepts, same alpha" — that is the plan's
   *active control* (§14.4 step 6: "Repeter avec des directions aleatoires comme controle
   actif"), not its *sham* (which is `k=0`, no injection at all — already covered by
   `--k_values` including 0). Rename `condition` values from `{"real", "sham"}` to
   `{"real", "random"}` so "sham" consistently means zero injections everywhere in the
   codebase, matching Exp 10's fix below.
5. Concept pool already has 10 concepts, comfortably above the "at least 4 distinct
   concepts" requirement — no change.

### Experience 10 — identify two injected concepts (`code/experiments/multi_identification.py`)

Current script implements the old Block 2 (`E2`/`C2.1-C2.3`). Required changes:

1. **Conditions must match §15.2's four rows exactly:** (a) one concept, (b) two concepts,
   (c) one concept + one random direction, (d) sham (zero injections). Today's
   `CANONICAL_CONDITIONS` has `c2_1_single` (a), `e2_free`/`c2_2_forced_choice` (b, split
   awkwardly by mode), and `c2_3_sham` (c, but named "sham" even though it's a random
   direction, not zero injections). There is **no true zero-injection condition** yet.
   Fixes:
   - Add a `true_sham` condition with `n_injections=0` (empty spec list).
   - Rename `c2_3_sham` → `concept_plus_random` and rename the `sham_slots` parameter/CLI
     flag to `random_slots` throughout, so "sham" is reserved for zero injections.
2. **Every condition runs in both response formats (§15.2, §15.5).** The plan evaluates all
   four conditions in forced-choice *and* free-response, as separate trials, forced-choice
   first so the candidate list can't contaminate the free-response wording (§15.4: "Elle
   vient apres le choix force dans des essais distincts afin que la liste de candidats ne
   contamine pas la reponse libre"). Today, mode is fixed per condition (forced-choice only
   for `c2_2_forced_choice`, free only for the rest). Restructure to a 4 (conditions) x 2
   (modes) matrix: for each sampled item (same drawn concepts/layers), run the forced-choice
   prompt first, then a second, independent generation call with the free-response prompt on
   the same injection — two trial records sharing an item id, sequenced in that order.
3. **Forced-choice format (§15.3).** Needs a `NONE` option per slot and an answer of
   "exactly two labels in alphabetical order," replacing today's open letter-set
   exact-match. Concretely:
   - `all_prompts.py:get_forced_choice_identification_messages` needs new wording matching
     §15.3, and the candidate list needs a `NONE` entry (its own letter, per the existing
     `CANDIDATE_LETTERS` scheme) alongside the concept distractors.
   - New parser in `response_parsing.py`, e.g. `parse_two_label_choice(text, valid_labels)`,
     tolerant of alphabetical-order formatting, returning up to two labels drawn from the
     concept letters plus `NONE`'s letter.
   - Grading in `identification_accuracy.py` needs a NONE-aware exact match: a random-slot's
     correct label is `NONE`, a real-concept slot's correct label is its assigned letter.
4. **Free-response format** stays close to today's `get_identification_messages` — just
   needs to be sequenced after forced-choice per item (point 2) rather than run standalone.
5. **Dosing.** Exp 10 only specifies one z per injection ("z identique pour chaque injection
   dans l'analyse principale") — no budget-constant regime here, unlike Exp 9.
6. **Layer pool** — same widening to all 32 layers as Exp 9.
7. **Distractors** — current code already builds distractors from the full 10-concept pool
   minus the injected concepts, satisfying §15.2's "regle de choix des distracteurs"
   requirement; just needs the `NONE` entry added alongside them.

### Experience 11 — layer order + modulators (`code/experiments/layer_ordering.py`, `code/experiments/modulators.py`)

Current scripts implement the old Block 3 (`E3`/`C3.1-C3.2`) and Block 4
(`E4`/`E5`/`E6`, the last two hard-restricted to identification). Required changes:

1. **Prompt (§16.2).** `get_ordering_messages` currently narrates "Between {concept_1} and
   {concept_2}, which one entered your processing first?" with the concept names filled
   directly into a sentence. Replace with the plan's `A) {CONCEPT_1}` / `B) {CONCEPT_2}`
   lettered format, answered with exactly one letter. This introduces a third
   counterbalancing axis that doesn't exist today: which concept is labeled `A` vs. `B` is
   independent of (i) which concept is actually shallow vs. deep, and (ii) the existing
   `presentation_order` (`shallow_first`/`deep_first`, which today conflates "order named in
   the sentence" with what should become "which letter it's given"). Add an independent
   random `A`/`B` assignment per trial, and a strict new parser (e.g.
   `parse_ab_choice(text) -> "A" | "B" | None`) replacing the current text-matching
   `parse_concept_choice`, since the model is now answering with a letter, not naming the
   concept.
2. **Modulators target ordering, not identification (§16.5).** The plan applies all four
   modulators — layer distance `|i-j|`, absolute position at fixed distance, dose ratio
   `z_A/z_B`, and concept-pair cosine similarity — to Exp 11 (ordering). It specifies no
   modulator sweep for Exp 10 at all. Today's `modulators.py` does the opposite for two of
   the three sweeps: `run_e5_alpha_ratio_sweep` and `run_e6_similarity_sweep` are hard-wired
   to identification (`get_identification_messages`, cosine-similarity scoring); only `E4`
   (distance/placement) currently runs both tasks. Required change:
   - Retarget `run_e5_alpha_ratio_sweep` and `run_e6_similarity_sweep` to use the new Exp 11
     ordering prompt/parser instead of identification, scoring by the A/B-letter `correct`
     field rather than concept-description cosine similarity.
   - Retarget `run_e4_distance_sweep` to ordering only (drop or flag-gate the identification
     arm, since it's no longer specified by the plan).
   - Keep the existing `--distances`/`--placements` defaults and the
     `_sample_layer_pair_with_distance` fallback for the extreme distances 1 and 31 — that
     logic already satisfies §16.5's "distances extremes 1 et 31 ... si les hooks le
     permettent."
3. **Layer pool** — `layer_ordering.py`'s default also needs widening to all 32 layers
   (`modulators.py`'s `ALL_LAYERS = list(range(0, 32))` is already correct).
4. **Presentation-order control (C3.1)** stays valid but is now a 2x2 with the new A/B-label
   assignment (order named x letter assigned), not a single axis.

## Cross-cutting changes

- **Layer pool default**: bump to `list(range(0, 32))` in `multi_detection.py`,
  `multi_identification.py`, `layer_ordering.py` (`modulators.py` already uses the full
  range). This roughly doubles the layer-combination space versus today's every-other-layer
  default; manage cost via `--num_trials` / sbatch time budgets rather than narrowing the
  pool, since the plan requires the full 32-layer draw.
- **"sham" naming**: reserve it for zero injections everywhere. `multi_detection.py`'s
  random-direction condition becomes `"random"`; `multi_identification.py`'s `sham_slots`
  becomes `random_slots`, freeing up `sham` for the new true-zero-injection condition.
- **Journaling (plan §18)**: no major gaps beyond what falls out of the changes above — dose
  regime, per-injection alpha, and label/order assignments just need to land in the trial
  records described per-experiment above. Existing records already capture layer, concept,
  vec_type, seed-derived direction identity, and presentation order.

## Consequences for analysis scripts and sbatch jobs

- `code/analysis/count_confusion_matrix.py`: add a `dose_regime` split so the confusion
  matrix and false-positive-rate tables can be reported separately for `individual` vs.
  `budget`.
- `code/analysis/identification_accuracy.py`: NONE-aware exact match for forced choice; a
  false-identification check for the new true-sham condition (parallel to the existing
  `sham_false_identification_rate`, which now applies to the renamed
  `concept_plus_random` condition instead).
- `code/analysis/ordering_accuracy.py`: switch from name-matched `reported_concept`/`correct`
  fields to the new letter-based `reported_letter`/`correct_letter`; add a label-assignment
  bias breakdown alongside the existing presentation-order bias breakdown.
- `code/analysis/modulator_plots.py`: `plot_e5`/`plot_e6` currently score via
  `identification_correct()` (cosine-similarity threshold) — switch to the ordering task's
  `correct` field. `plot_e4` drops its identification panel (or keeps it behind a flag) once
  `run_e4_distance_sweep` is ordering-only.
- `jobs/multi_detection.sbatch`: add `--dose_regime` / `--z_total` (run `individual` and
  `budget` as two submissions, or loop both inside one job); layer pool widens automatically
  once the script's default changes (no `--layer_pool` flag is currently passed).
- `jobs/multi_identification.sbatch`: `--run_all_conditions` needs to pick up the new
  4x2 condition/mode matrix; downstream, `identification_accuracy.py --inputs` default list
  needs updating to the new set of output filenames.
- `jobs/layer_ordering.sbatch`: no flag changes required — picks up the new prompt/parser and
  wider layer pool via the script's own defaults.
- `jobs/modulators.sbatch`: `--experiments e4_distance e5_alpha_ratio e6_similarity` flag
  names stay stable even though `e5_alpha_ratio`/`e6_similarity` now target ordering instead
  of identification — no CLI change needed, only the scripts' internals.

## Explicitly out of scope for this pass

- Experience 0 (`s(l,v)` natural-scale calibration, SD/MAD) — not implemented; `z` is used
  directly as the injection amplitude, per the scope decision above.
- Experiences 1-8 (single-injection 2AFC localization, presence detection, alpha-vs-z
  comparison, the §17.6 pilot) — separate workstream.
- `code/experiments/calibration.py` / `jobs/calibration.sbatch` (old Block 0) — left as-is;
  no longer a required gate before Exp 9-11, since there's no scale-derived window to
  compute from it.

## Next steps

1. Update `all_prompts.py`: rewrite `get_count_report_messages`, the forced-choice
   identification prompt (+ `NONE` handling), and `get_ordering_messages` (A/B lettered) to
   match §14.3 / §15.3 / §16.2 verbatim.
2. Extend `response_parsing.py`: add `parse_ab_choice` (Exp 11) and
   `parse_two_label_choice` (Exp 10, letters + `NONE`); `parse_count` needs no change.
3. Update `multi_detection.py`: add `--dose_regime {individual,budget}` + `--z_total`, widen
   the default layer pool to 32, rename `condition="sham"` → `"random"`.
4. Update `multi_identification.py`: add the true zero-injection sham condition, cross all
   four conditions with both response formats (forced-choice before free-response per item),
   rewire forced-choice to the `NONE`/alphabetical-order format, rename
   `sham_slots`→`random_slots`, widen the layer pool.
5. Update `layer_ordering.py`: add independent A/B label counterbalancing, new prompt and
   parser, widen the layer pool.
6. Update `modulators.py`: retarget `E5`/`E6` (and `E4`'s primary arm) from identification to
   ordering, using the new Exp 11 prompt/parser.
7. Update the four analysis scripts (`identification_accuracy.py`, `ordering_accuracy.py`,
   `modulator_plots.py`, `count_confusion_matrix.py`) for the record-schema changes above.
8. Update `jobs/multi_detection.sbatch` and `jobs/multi_identification.sbatch` for the new
   CLI flags / output filenames; `layer_ordering.sbatch` and `modulators.sbatch` need no flag
   changes.
9. Smoke-test all four scripts with a small `--num_trials` locally/interactively before
   resubmitting full sbatch runs.
10. Resubmit `multi_detection`, `multi_identification`, `layer_ordering`, `modulators` on the
    cluster, then rerun the updated analysis scripts.
