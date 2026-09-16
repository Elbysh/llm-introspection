# LLM Introspection

Research code for a computational-psychophysics study of **behavioral introspection in LLMs**: How does a language model react to different types of activation perturbations ? The project builds a controlled protocol (activation steering + self-report), aiming to disentangle genuine introspection from confounds such as anomaly detection, response bias, or plain model degradation.

Full experimental design (in French): [`docs/livrables/cadrage-experiments.md`](docs/livrables/cadrage-experiments.md).

## Core idea

Activations are perturbed at a chosen Transformer layer and the model is asked to report on that perturbation (which sentence was targeted, whether *any* intervention occurred, etc.). Four perturbation families are compared:

- **Concept** — a unit direction built from contrastive prompts (e.g. `Dust`, `betrayal`, `recursion`)
- **Fixed random** — a fixed random unit direction (no semantic content)
- **Noise** — a random direction redrawn at every token
- **Dropout** — multiplicative Bernoulli corruption of the activation itself

Each is measured under two dose parametrizations: raw amplitude **α** and amplitude standardized by the natural variability of activations, **z = α / s(layer, direction)**. The central question is whether the ranking of "how detectable" each family is changes once amplitude is expressed relative to natural activation variance rather than in raw units.

Primary model: `meta-llama/Llama-3.1-8B-Instruct`, with `Qwen/Qwen3.8-27B` as a secondary model once protocols are validated.

## Experiments

| # | Name | Question |
|---|------|----------|
| 0 | Calibration | Estimate the natural activation scale `s(layer, direction)` used to convert α ↔ z |
| 1 | Localization / psychometrics | 2AFC "which sentence was targeted", swept over α and z for all four families |
| 2 | Presence detection | Can the model tell a real perturbation from a sham (no-op) intervention, controlling for response bias (d′, criterion, AUROC) |
| 3 | Variability | How much does detectability vary across concepts, directions, and layers |
| 4 | Task degradation | Does detection persist where the model still performs a normal semantic task, or only once it's already broken |
| 5 | Text induction | Does text merely *evocative* of a concept (no activation change) increase false-positive intervention reports |
| 6–9 | Multi-injection extensions | Confidence, counting multiple injections, identifying two concepts, ordering injected layers |

Each experiment lives under `code/experiment_<n>_<name>/` (currently fully built out for experiment 0 and experiment 5) or as a flatter module under `code/experiments/` (experiments 1–4 and the multi-injection extensions). Every experiment folder has its own `README.md` with exact commands.

## Repository layout

```text
code/
  experiment_0_calibration/   step-by-step natural-scale calibration pipeline
  experiment_5_text_induction/  text-induction false-positive pipeline
  experiments/                 experiment 1-4 + multi-injection scripts, shared entrypoints
  analysis/                    plotting and accuracy/report scripts used across experiments
  calibration/                 standalone directional-scale calibration module
  utils/                       activation injection, hooks, prompts, model registry, judging
configs/            YAML/JSON run configs per experiment
data/               concept datasets, saved direction vectors, calibration contexts
jobs/                SLURM (.sbatch) job scripts for running experiments on the Ruche cluster
results/             per-experiment outputs: metrics, figures, manifests
docs/
  livrables/          experimental design & framing documents (deliverables)
  misc/                findings write-ups, LaTeX report drafts, implementation notes
  presentation/        slides
  results.tex, proposed_approach.tex, appendix_*.tex  paper/report sources
tests/               pytest suites (experiment 0, experiment 5, experiment 1 end-to-end)
logs/                SLURM stdout/stderr logs from past cluster runs
```

## Getting started

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (see `pyproject.toml` / `uv.lock`); Python ≥ 3.13.

```bash
uv sync
```

Example: run the experiment 0 calibration pipeline locally, step by step:

```bash
python -m experiment_0_calibration.prepare_concept_vectors --config configs/experiment_0_calibration/development_full.yaml
python -m experiment_0_calibration.prepare_material_plan   --config configs/experiment_0_calibration/development_full.yaml
python -m experiment_0_calibration.run_experiment_0         --config configs/experiment_0_calibration/development_full.yaml
```

Or on the Ruche cluster:

```bash
bash jobs/experiment_0_calibration/submit.sh development_full
```

Run tests with:

```bash
python -m pytest tests -q
```

See the per-experiment `README.md` files (e.g. [`code/experiment_0_calibration/README.md`](code/experiment_0_calibration/README.md), [`code/experiment_5_text_induction/README.md`](code/experiment_5_text_induction/README.md)) for exact commands, config format, and produced artifacts.
