# Multisteering results

These are the saved outputs for Experiences 9–11 of the multisteering study.
Raw trial tensors are kept next to the experiment that produced them; report
figures are shared because the report combines all three experiments.

```text
multisteering/
├── experiment_9_counting/raw/
├── experiment_10_identification/raw/
├── experiment_11_ordering/raw/
├── experiment_11_modulators/raw/
└── figures/
```

The original Round 1 tensors are preserved under the relevant
`raw/archive_alpha1-7/` directory. Figures can be regenerated from the raw
files with:

```bash
.venv/bin/python docs/misc/reports/generate_figures.py
.venv/bin/python docs/misc/reports/generate_bias_control_figures.py
```
