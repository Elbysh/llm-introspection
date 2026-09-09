# Next Steps

1. **Get `calibration.sbatch` to actually complete.** Submit it for real via `sbatch`
   (not interactive) once the import-speed issue is confirmed resolved on a second
   run. Produces `plots/calibration_trials.pt`.

2. **Run `calibration_accuracy.py` on that output.** This is the go/no-go gate from
   the plan: it picks the (layer, alpha) operating window where detection rate is
   reliably above the alpha=0 false-positive rate. Every Block 1-4 script currently
   defaults to `--alpha 8.0` as a placeholder that needs replacing with whatever
   calibration actually shows.

3. **Update the Block 1-4 sbatch scripts' `--alpha`** (and possibly narrow
   `--layer_pool` / `--layers` / `--distances` if calibration shows some layers
   just don't work at all) based on that operating window, before submitting
   anything else.

4. **Submit Blocks 1-3** (independent of each other, order doesn't matter much):
   - `multi_detection.sbatch` -> `count_confusion_matrix.py`
   - `multi_identification.sbatch` -> `identification_accuracy.py`
   - `layer_ordering.sbatch` -> `ordering_accuracy.py`

5. **Submit `modulators.sbatch`** (Block 4) last — heaviest run (E4+E5+E6 combined,
   ~650 generations), and depends on Block 2/3 already having sane-looking results
   since it's parametrized sweeps of the same tasks. Then `modulator_plots.py`.

6. **Sanity-check the identification similarity threshold before trusting Block 2
   results.** `identification_accuracy.py`'s embedding-similarity threshold (0.5) is
   currently a guess, not pre-registered against data. Pull the alpha=0
   (no-injection) responses from calibration's output and check what similarity
   score those get against random concept descriptions, so 0.5 isn't accidentally
   crossed by chance/generic text.
