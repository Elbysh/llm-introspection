# Multi-Steering: Adapt Code & Build Experiment Pipeline

## Context

`plan_of_research.md` (already on this branch, `louis/multisteering`) lays out a 5-block
experimental program testing whether Llama-3.1-8B-Instruct can introspect on **multiple
simultaneous concept injections at different layers** — extending the repo's existing
single-concept work (Lindsey 2025 replication in `code/experiments/position_detection.py`
and `strength_comparison.py`).

The current codebase only injects **one vector at one layer** at a time
(`code/utils/inject_concept_vector.py`), or one vector at **two token positions within a
single layer** (`strength_comparison.py`'s `make_dual_injection_hook`). Nothing injects
different vectors at different *layers* simultaneously, which is the core mechanic every
block of the plan needs. There's also no sham/control-vector generator, no free-response
grading, and no confusion-matrix / distance-sweep analysis — all required by Blocks 1-4.

This was planned in a sandbox with no GPU, no `torch`, and no SLURM client, so nothing was
executed there. This pass produces **code + sbatch jobs** to submit on the cluster
(`prod40`/`prod80`, matching the existing `jobs/*.sbatch` pattern) — not live experiment
runs. Grading uses **local embedding similarity** (no LLM-judge API), and the sham/hook
infra is written **fresh** rather than pulling in `origin/feat/random_vectors` /
`origin/feat/gaussian_noise_dropout_hooks`.

Given the size of the plan, Phase 1 below delivers the **shared multi-injection
infrastructure + Block 0 (calibration)**, meant to be validated end-to-end before Blocks
1-4 are built on top of it as follow-up work, so the core injection engine doesn't need to
be redesigned mid-way.

---

## Phase 1: Core infra + Block 0

### 1. `code/utils/multi_inject.py` — the engine every later block reuses

Generalizes `inject_concept_vector.py`'s single-hook pattern to **K simultaneous
injections at arbitrary (possibly repeated) layers**:

```python
@dataclass
class InjectionSpec:
    layer: int
    vector: torch.Tensor      # will be unit-normalized internally
    alpha: float
    token_range: tuple | None = None   # None = all tokens, prompt + every generated token

@contextmanager
def apply_multi_injection(model, specs: list[InjectionSpec]):
    ...
```

- Groups specs by `layer`, registers **one forward hook per distinct layer** on
  `model.model.layers[l]`, summing each spec's `alpha * unit(vector)` contribution —
  handles the (rare but real) case of two injections landing on the same layer.
- Default `token_range=None` reproduces `inject_concept_vector.py`'s "inject at all
  tokens" behavior, which — because hooks fire on every incremental decode step too —
  keeps the injection *sustained through generation*, matching the plan's prompts
  ("I may or may not have injected ... while processing this message").
- Optional `token_range` per spec reuses the position-specific pattern from
  `position_detection.py` / `strength_comparison.py`, for later logit-lens /
  non-verbal-readout cross-checks (cross-cutting validity check #2).
- Context manager guarantees hook removal even on exception (existing scripts do manual
  `handle.remove()`, which leaks hooks on error paths).

### 2. `code/utils/sham_vectors.py` — control vectors for C1.4 / C2.3

`make_random_direction(hidden_dim, seed) -> torch.Tensor`: deterministic, unit-norm,
isotropic random direction (`torch.Generator().manual_seed(seed)`), generated on the fly
per trial — no disk precompute needed (unlike real concept vectors, a sham direction
carries no semantics to cache). `hidden_dim` inferred from an existing file under
`data/saved_vectors/llama` (same trick as reading any saved concept vector's shape), so it
never needs hardcoding to 4096.

### 3. `code/utils/response_parsing.py` — deterministic label extraction

Since grading is local/embedding-based rather than an LLM judge, free-text responses still
need robust extraction of the structured part of the answer:
- `parse_yes_no(text) -> bool | None`
- `parse_count(text, max_k=4) -> int | None` (for E1, later)
- `parse_scale_0_10(text) -> int | None` (for C1.3, later)
- `is_coherent(text) -> bool`: cheap heuristic filter (min length, not degenerate/repeated
  tokens, actually answers) standing in for the paper's `coherence_prompt` LLM check —
  flagged and excluded from accuracy denominators, with the exclusion rate reported.

### 4. `code/utils/embedding_judge.py` — local grading

Thin wrapper around `sentence-transformers` (new dependency): `embed(text)`,
`cosine_similarity(response_text, concept_description) -> float`. Used in calibration to
cross-check that an affirmative "yes I notice something" is *concept-specific* and not
generic yes-bias, and reused as-is for Block 2's free-response grading later.

### 5. `code/utils/concepts.py` — concept registry (minimal now, extended later)

One-line canonical descriptions for the 10 existing concepts (5 from `complex_data.json`:
`fibonacci_numbers, recursion, betrayal, appreciation, shutdown`; 5 from
`simple_data.json`: `Dust, Satellites, Trumpets, Origami, Illusions`), used by
`embedding_judge.py`. Kept small and separate from prompt templates so Block 2's
candidate-list / cosine-similarity-bucket helpers (Block 4/E6) can extend it later without
touching calibration.

### 6. `code/experiments/calibration.py` — Block 0

Single-injection sweep using `apply_multi_injection` with a single `InjectionSpec`
(K=1 is just the general engine's base case — this is also the first real exercise of the
engine before Blocks 1-4 stack multiple specs on it):

- Sweep `layer ∈ {0,2,...,30}` × `alpha ∈ {0, 1, 2, 4, 6, 8, 12, 16}` (α=0 doubles as the
  null/false-positive condition — no separate no-injection branch needed).
- Concept for each trial drawn randomly from the 10-concept pool (cross-cutting validity
  check #4: randomize concept↔layer assignment).
- Prompt: the plan's exact calibration prompt via `tokenizer.apply_chat_template`.
- `model.generate(..., do_sample=False)` with injection sustained throughout (per engine
  default).
- Per trial, record: raw response text, `is_coherent`, `parse_yes_no` (did it claim
  noticing something), `cosine_similarity(response, concept_description)` (is it the
  *right* concept, not just generic anomaly-detection).
- Save trial-level records as `.pt` (same convention as `strength_comparison.py`), one file
  per run.

### 7. `code/analysis/calibration_accuracy.py`

Loads calibration's saved trials, produces: detection-rate and false-positive-rate
(α=0 rows) table by `(layer, alpha)`, a similarity-vs-baseline check, and a heatmap plot
(reusing the `matplotlib` conventions in `plot_strength_comparison_adjusted.py`) — this is
literally "select operating window for Blocks 1-4" from the plan.

### 8. `jobs/calibration.sbatch`

Same shape as `jobs/strength_comparison.sbatch` (source `.venv`, `cd data && PYTHONPATH=../code/utils uv run python ../code/experiments/calibration.py ...`, `prod40`, 4h).

### 9. `pyproject.toml`

Add `sentence-transformers` (embedding grading) and `scipy` (binomial tests needed
starting Block 3's C3.2 — adding now avoids a second dependency-touching PR).

---

## Phase 2 (follow-up): Blocks 1-4

Recorded here so the file layout is planned ahead, not built in Phase 1:

- `code/experiments/multi_detection.py` (Block 1, E1 + C1.1-C1.4) — first real multi-spec
  use of `apply_multi_injection` (K ∈ {0..4} distinct layers), sham specs via
  `sham_vectors.py` for C1.4.
- `code/experiments/multi_identification.py` (Block 2, E2 + C2.1-C2.3) — free response
  graded via `embedding_judge.py`; forced-choice (C2.2) via `response_parsing`'s
  letter-parser (new, small addition) against a randomized 10-concept candidate list from
  `concepts.py`.
- `code/experiments/layer_ordering.py` (Block 3, E3 + C3.1-C3.2) — 2AFC "first/second"
  parser, `scipy.stats.binomtest` vs 50%.
- `code/experiments/modulators.py` (Block 4, E4/E5/E6) — thin parametrized wrappers around
  the Block 2/3 scripts sweeping `|layer_i - layer_j|`, `alpha_i/alpha_j`, and a
  cosine-similarity-bucketed concept-pair selector (new helper in `concepts.py` that reads
  `data/saved_vectors/llama` directly, no new precompute).
- `code/analysis/count_confusion_matrix.py`, `identification_accuracy.py`,
  `ordering_accuracy.py`, `modulator_plots.py` — one analysis script per block, following
  the existing `compute_position_detection_accuracy.py` / `generate_adjusted_accuracy_table.py` pattern.
- Matching `jobs/*.sbatch` per script.

---

## Verification plan (for whoever implements Phase 1)

No real model is available in the environment this plan was written in, so implementation
should:
1. **Static review** of `multi_inject.py` against the existing single-injection behavior in
   `inject_concept_vector.py` and the dual-position hook in `strength_comparison.py` —
   confirm the K=1, single-layer case reduces to identical tensor math.
2. **Synthetic hook test** (`code/utils/test_multi_inject.py`): a tiny fake decoder stack
   (a few `nn.Module`s returning `(hidden_states,)` tuples, small hidden dim, no HF
   download) exercising `apply_multi_injection` with multiple specs — asserts per-layer
   additive combination, correct broadcasting, and hook cleanup on exception. This needs
   only CPU `torch`, not the 8B model, so it can run without cluster access.
3. Submit `jobs/calibration.sbatch` on the cluster; `calibration_accuracy.py`'s output
   table is the actual go/no-go signal for picking Blocks 1-4's operating window before
   writing Phase 2.
