"""Clean-sham bias audit; resample prompts, never intervention references."""
from pathlib import Path
import numpy as np
import pandas as pd


def unique_shams(frame, columns):
    """Reject inconsistent cached values before removing repeated references."""
    if frame.groupby('sham_trial_id')[columns].nunique(dropna=False).gt(1).any().any():
        raise ValueError('Conflicting values for a cached sham')
    return frame.drop_duplicates('sham_trial_id')


def cluster_estimates(frame, metrics, cluster, iterations, rng):
    # Each cluster retains both mappings and all order/label variants.
    grouped = frame.groupby(cluster)[metrics]
    sums = grouped.sum().to_numpy(float)
    sizes = grouped.size().to_numpy(float)
    indices = rng.integers(0, len(sizes), size=(iterations, len(sizes)))
    draws = sums[indices].sum(axis=1) / sizes[indices].sum(axis=1)[:, None]
    return frame[metrics].mean(), np.quantile(draws, [0.025, 0.975], axis=0), draws


def generate_controls(root: Path, output: Path, iterations=2000, seed=20260914):
    if iterations < 2:
        raise ValueError('At least two bootstrap replicates are required')
    rng = np.random.default_rng(seed)
    results, all_draws = [], []
    for directory in sorted(root.glob('layer_*'), key=lambda p: int(p.name.split('_')[-1])):
        if not (directory / 'summary.json').exists():
            continue
        layer = int(directory.name.split('_')[-1])
        pcols = ['kind', 'sham_trial_id', 'pair_id', 'mapping', 'score', 'finite']
        pieces = [c[c.kind.eq('sham')] for c in pd.read_csv(directory / 'trials.csv', usecols=pcols, chunksize=100000)]
        presence = unique_shams(pd.concat(pieces), pcols)
        for task, data, cluster, scorecol, finitecol in [
            ('presence', presence, 'pair_id', 'score', 'finite'),
        ]:
            data = data.copy()
            finite = data[finitecol].astype(str).str.lower().eq('true') & np.isfinite(data[scorecol])
            n_excluded = int((~finite).sum())
            data = data[finite].copy()
            if data.empty:
                raise ValueError(f'No finite {task} shams in {directory}')
            score = data[scorecol]
            decision = (score > 0).astype(float) + 0.5 * score.eq(0)
            data['mean_margin'] = score
            data['tie_rate'] = score.eq(0).astype(float)
            data['false_alarm_rate'] = decision
            yes = decision
            metrics = ['false_alarm_rate', 'x_choice_rate', 'mean_margin', 'tie_rate']
            data['x_choice_rate'] = np.where(data.mapping.eq('XY'), yes, 1-yes)
            for mapping in ['pooled', 'XY', 'YX']:
                subset = data if mapping == 'pooled' else data[data.mapping.eq(mapping)]
                if subset.empty:
                    continue
                estimates, ci, draws = cluster_estimates(subset, metrics, cluster, iterations, rng)
                for j, metric in enumerate(metrics):
                    identity = dict(layer=layer, task=task, mapping=mapping, metric=metric)
                    results.append(dict(**identity, estimate=estimates[metric], ci_low=ci[0,j], ci_high=ci[1,j], n_shams=len(subset), n_clusters=subset[cluster].nunique(), n_excluded_task=n_excluded, bootstrap_iterations=iterations, seed=seed))
                    all_draws.append(pd.DataFrame(dict(**identity, replicate=np.arange(iterations), value=draws[:,j])))
    table = pd.DataFrame(results)
    table.to_csv(output / 'controls_bias.csv', index=False)
    pd.concat(all_draws).to_csv(output / 'controls_bootstrap_draws.csv.gz', index=False, compression='gzip')
    lines = ['# Clean controls: baseline model bias', '',
             f'Pointwise 95% percentile intervals from {iterations:,} prompt-cluster bootstrap replicates (seed {seed}). The bootstrap resamples sentence pairs. Mappings and order variants stay together in pooled estimates. Cached shams are counted once per layer. Layers repeat the same clean panel and are not independent replications.', '',
             'Positive margin means intervention. Ties contribute 0.5 to decisions. A zero-width interval at a boundary reflects no variation in this small panel, not certainty about new prompts. These intervals condition on the selected prompts/concepts and model.', '',
             '| Layer | Task | Mapping | Metric | Estimate | 95% CI | Unique shams | Prompt clusters |',
             '|---:|---|---|---|---:|---|---:|---:|']
    identical = table.groupby(['task', 'mapping', 'metric']).estimate.nunique().eq(1).all()
    display = table
    if identical:
        display = table[table.layer.eq(table.layer.min())]
        lines.insert(4, f'All {table.layer.nunique()} layers have identical baseline point estimates. The compact table uses layer {table.layer.min()} and its bootstrap intervals; the CSV contains every layer. Differences in interval endpoints across layers are bootstrap Monte Carlo variation.\n')
    for r in display.itertuples():
        lines.append(f'| {r.layer} | {r.task} | {r.mapping} | {r.metric} | {r.estimate:.3f} | [{r.ci_low:.3f}, {r.ci_high:.3f}] | {r.n_shams} | {r.n_clusters} |')
    (output / 'controls_bias.md').write_text('\n'.join(lines)+'\n')
    return table
