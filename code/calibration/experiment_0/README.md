# Experiment 0 — natural directional-scale calibration

This directory implements section 4, “Expérience 0”, of
`docs/propositions/cadrage-experiments.md`. It does not implement behavioural
trials, injections, or the later conversion of standardized doses into
coefficients `alpha`.

## Protocol-to-code map

| Protocol step | Implementation |
| --- | --- |
| Material: 100 distinct sentences | `plan.load_protocol_corpus` |
| Material: exact contexts and admissible positions | `plan.build_context_and_observation_rows` |
| Material: concept, fixed-random and renewed-noise directions | `plan.build_direction_rows` |
| 1. Natural forward passes | `run.collect_activations` |
| 2. Extract `h(block, token)` | `run.collect_activations` |
| 3. Compute `<h, v>` | `run.calibrate_layer` |
| 4. Mean, SD, median, corrected MAD, quantiles | `statistics.summarize_projection_matrix` |
| 5. Distribution and diagnostic plots | `plots.create_all_plots` |
| 6. Phrase-cluster bootstrap | `statistics.build_bootstrap_plan` |
| 7. Freeze `s_SD` | enforced by immutable revisions and output no-overwrite rules when status is `frozen` |
| 8. Convert `z` to `alpha` | intentionally deferred |

## Important conventions

- `decoder_block_index` always means the output hook on
  `model.model.layers[index]`.
- Existing concept artifacts are indexed like Hugging Face hidden states.
  Therefore decoder-block output `l` uses concept artifact `l + 1`.
- Point estimates give equal weight to every admissible token. Phrase-cluster
  bootstrap resamples whole sentences and preserves all their tokens.
- `fixed_random` directions remain fixed by identifier.
- `renewed_noise` receives a distinct Gaussian direction for every planned
  token and repetition. It is not a second fixed random bank.
- Seeds, source vector information, contexts, tokens, bootstrap settings,
  software version and file hashes are retained.

## Development versus frozen calibration

`configs/calibration/experiment_0/development_full.yaml` is intentionally not a
frozen calibration. It uses isolated sentences because the exact behavioural
presentation context and final number of repeated noise trials are not fixed
yet. The resulting plan is nevertheless explicit and auditable.

Before changing `protocol.status` to `frozen`:

1. replace the presentation template with the exact retained experimental
   context, switch `presentation.mode` to `external_manifest`, and provide a
   JSONL file whose rows contain `context_id`, `rendered_text`, and target
   `{sentence_id, char_start, char_end}` spans;
2. set the required number of renewed-noise repetitions per token;
3. replace model and tokenizer `main` revisions with immutable commit hashes;
4. record and approve the development/hold-out concept split.

The configuration validator rejects a nominally frozen configuration that
still uses moving model revisions or a development context.

## Commands

```bash
python -m calibration.experiment_0.ensure_concept_vectors \
  --config configs/calibration/experiment_0/development_full.yaml

python -m calibration.experiment_0.prepare_plan \
  --config configs/calibration/experiment_0/development_full.yaml

python -m calibration.experiment_0.run \
  --config configs/calibration/experiment_0/development_full.yaml
```

On Ruche, submit the dependent jobs with:

```bash
bash jobs/calibration/experiment_0/submit.sh development_full
```

## Outputs

The plan contains `contexts.jsonl`, `observations.jsonl`, `directions.jsonl`
and `manifest.json`. Results contain:

- `directional_scales.json` and `.csv` with one traceable record per
  `(decoder block, direction)`;
- `projections.npz`, aligned with `observations.jsonl` rather than anonymously
  flattened;
- `run_manifest.json` with provenance and SHA-256 hashes;
- primary SD curves, SD/MAD sensitivity, bootstrap stability and standardized
  projection-distribution heatmaps covering every block and family.

An existing output directory is never overwritten.
