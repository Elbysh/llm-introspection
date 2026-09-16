# Natural directional-scale calibration

This module estimates the natural activation scale
`s(layer, direction)` needed to express later intervention amplitudes as
`z = alpha / s(layer, direction)`. It is deliberately independent of the 2AFC
task: it neither builds sentence pairs nor injects activations.

## What is calibrated

For every configured decoder layer and unit direction `v`, the code projects
all non-padding token activations from the repository's existing
`LOCALIZATION_SENTENCES` corpus:

```text
q = v^T h(layer, token)
```

It saves the sample standard deviation of `q` and a Gaussian-consistent robust
alternative, `MAD = 1.4826 median(|q - median(q)|)`. Concept directions reuse
the vectors already present in `data/saved_vectors/llama`. Random and noise
directions are reproducible unit Gaussian draws.

Dropout has no direction to project onto, so it is calibrated differently: the
run also records the activation norms of every layer. An experiment converts a
target amplitude into a rate with `p = rho^2 / (1 + rho^2)`, `rho = alpha /
rms_norm(layer)`, and standardizes it against the median scale of the random
bank at that layer.

## Entry points

- `python -m calibration.directions --config configs/calibration/pilot.yaml`
  builds the normalized direction bank.
- `python -m calibration.compute_scales --config configs/calibration/pilot.yaml`
  runs the model, saves the calibration, and creates plots.
- `bash jobs/submit_calibration.sh pilot` submits both steps on Ruche.

Use `full` instead of `pilot` for every configured concept and more control
directions.

## Outputs

Each run writes only under `results/calibration/<run>/`:

- `calibration.json`: one auditable record per `(layer, direction)`;
- `calibration_table.csv`: the same summary in tabular form;
- `calibration_projections.npz`: scalar projections used to recompute or audit
  SD, MAD, quantiles and distributions;
- `calibration_layer_norms.json`: one record per layer with the RMS, mean,
  median and tail quantiles of the token activation norms, the scale a
  direction-free perturbation such as dropout is matched against;
- `figures/directional_scale_by_layer.png`: linear scale comparison;
- `figures/directional_scale_by_layer_log.png`: the same comparison on a log
  axis, useful when concept scales dominate controls;
- `figures/sd_vs_mad.png` and `figures/sd_over_mad_by_layer.png`: robustness and
  tail diagnostics;
- `figures/projection_distributions.png`: representative standardized shapes.

The saved full hidden-state tensors are not retained; only one-dimensional
projections are kept, which is sufficient for this calibration and much
lighter.
