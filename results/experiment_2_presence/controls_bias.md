# Clean controls: baseline model bias

Pointwise 95% percentile intervals from 2,000 prompt-cluster bootstrap replicates (seed 20260914). The bootstrap resamples sentence pairs. Mappings and order variants stay together in pooled estimates. Cached shams are counted once per layer. Layers repeat the same clean panel and are not independent replications.

All 31 layers have identical baseline point estimates. The compact table uses layer 0 and its bootstrap intervals; the CSV contains every layer. Differences in interval endpoints across layers are bootstrap Monte Carlo variation.

Positive margin means intervention. Ties contribute 0.5 to decisions. A zero-width interval at a boundary reflects no variation in this small panel, not certainty about new prompts. These intervals condition on the selected prompts/concepts and model.

| Layer | Task | Mapping | Metric | Estimate | 95% CI | Unique shams | Prompt clusters |
|---:|---|---|---|---:|---|---:|---:|
| 0 | presence | pooled | false_alarm_rate | 0.000 | [0.000, 0.000] | 40 | 5 |
| 0 | presence | pooled | x_choice_rate | 0.500 | [0.500, 0.500] | 40 | 5 |
| 0 | presence | pooled | mean_margin | -3.734 | [-3.928, -3.547] | 40 | 5 |
| 0 | presence | pooled | tie_rate | 0.000 | [0.000, 0.000] | 40 | 5 |
| 0 | presence | XY | false_alarm_rate | 0.000 | [0.000, 0.000] | 20 | 5 |
| 0 | presence | XY | x_choice_rate | 0.000 | [0.000, 0.000] | 20 | 5 |
| 0 | presence | XY | mean_margin | -3.431 | [-3.619, -3.244] | 20 | 5 |
| 0 | presence | XY | tie_rate | 0.000 | [0.000, 0.000] | 20 | 5 |
| 0 | presence | YX | false_alarm_rate | 0.000 | [0.000, 0.000] | 20 | 5 |
| 0 | presence | YX | x_choice_rate | 1.000 | [1.000, 1.000] | 20 | 5 |
| 0 | presence | YX | mean_margin | -4.037 | [-4.244, -3.869] | 20 | 5 |
| 0 | presence | YX | tie_rate | 0.000 | [0.000, 0.000] | 20 | 5 |
