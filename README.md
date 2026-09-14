
### Experiment 2: intervention detection

The standalone detection runner is `code/experiments/experiment2_presence.py`, with
its SLURM job in `jobs/experiment2.sbatch`. The [protocol and results](docs/livrables/experiment2-detection.md)
include bootstrap uncertainty, clean-sham bias, output-token diagnostics and alpha-to-z plots.
The compact [published snapshot](docs/results/experiment2/) redraws without model inference:

```sh
python code/analysis/plot_experiment2.py docs/results/experiment2 --output-dir /tmp/experiment2-figures
```
