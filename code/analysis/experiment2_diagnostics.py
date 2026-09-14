"""Dose conversion and unrestricted-token diagnostics for the locked run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FAMILIES = ('concept', 'random', 'noise', 'dropout')
LABELS = ('Concept', 'Fixed random', 'Renewed noise', 'Dropout')


def alpha_to_z(alpha, scale):
    scale = np.asarray(scale, dtype=float)
    if np.any(~np.isfinite(scale)) or np.any(scale <= 0):
        raise ValueError('Calibration scales must be finite and positive')
    return np.asarray(alpha, dtype=float) / scale


def conversion_table(root, calibration_csv):
    scales = pd.read_csv(calibration_csv, low_memory=False).set_index('direction_id', drop=False)
    rows = []
    for path in sorted(root.glob('layer_*/manifest.json')):
        manifest = json.loads(path.read_text())
        layer = int(path.parent.name.split('_')[-1])
        alphas = sorted({p['dose'] for p in manifest['plan'] if p['matching'] == 'alpha'})
        local = scales[scales.decoder_block_index.eq(layer)]
        for family in FAMILIES:
            if family in ('concept', 'random'):
                ids = sorted({p['direction_id'] for p in manifest['plan'] if p['family'] == family})
                group = local.loc[ids]
            elif family == 'noise':
                group = local[local.direction_family.eq('renewed_noise')]
            else:
                group = local[local.direction_family.eq('fixed_random')]
            if group.empty:
                raise ValueError(f'No scales for {layer}/{family}')
            for estimator, field in [('sd', 'sd'), ('mad', 'mad_corrected')]:
                values = group[field]
                if family == 'dropout':
                    values = pd.Series({'fixed_random_bank_median': values.median()})
                for direction, scale in values.items():
                    for alpha in alphas:
                        rows.append(dict(layer=layer, family=family, direction_id=direction,
                                         estimator=estimator, scale=scale, alpha=alpha,
                                         z=float(alpha_to_z(alpha, scale))))
    return pd.DataFrame(rows)


def plot_conversions(table, output):
    layers = sorted(table.layer.unique())
    selected = [v for v in (0, 5, 13, 20, 30) if v in layers] or layers[:5]
    fig, axes = plt.subplots(2, 4, figsize=(17, 9), layout='constrained')
    for row, estimator in enumerate(('sd', 'mad')):
        for col, family in enumerate(FAMILIES):
            ax = axes[row, col]
            for layer in selected:
                part = table[table.layer.eq(layer) & table.family.eq(family) & table.estimator.eq(estimator)]
                stats = part.groupby('alpha').z.agg(['median', 'min', 'max'])
                line, = ax.plot(stats.index, stats['median'], label=f'L{layer}', lw=1.7)
                ax.fill_between(stats.index, stats['min'], stats['max'], color=line.get_color(), alpha=.10)
            ax.set(xscale='log', yscale='log', xlabel='Requested raw amplitude α',
                   ylabel=f'z ({estimator.upper()})', title=LABELS[col])
            ax.axvline(128, color='black', ls=':', lw=.8)
            ax.axhline(1, color='gray', ls=':', lw=.8)
            ax.grid(alpha=.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=len(selected), frameon=False)
    fig.suptitle('Raw amplitude to calibrated dose: z = α / scale\nLines: median z across directions; shading: full direction range, not a confidence interval', fontsize=14)
    fig.savefig(output / 'fig08_alpha_to_z.png', dpi=180)
    plt.close(fig)
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), layout='constrained')
    for row, estimator in enumerate(('sd', 'mad')):
        for col, family in enumerate(FAMILIES):
            ax = axes[row,col]
            part = table[table.alpha.eq(128) & table.family.eq(family) & table.estimator.eq(estimator)]
            stats = part.groupby('layer').z.agg(['median','min','max'])
            ax.plot(stats.index, stats['median'], color='#2563eb')
            ax.fill_between(stats.index, stats['min'], stats['max'], color='#2563eb', alpha=.15)
            ax.axhline(20.48, color='#b45309', ls='--', lw=1, label='Largest tested z = 20.48')
            ax.set(yscale='log', xlabel='Injection layer', ylabel=f'z ({estimator.upper()})', title=LABELS[col])
            ax.grid(alpha=.2)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', frameon=False)
    fig.suptitle('What α = 128 means at each layer\nMedian and full direction range; dropout uses the fixed-random bank median scale', fontsize=14)
    fig.savefig(output / 'fig09_alpha128_equivalent_z.png', dpi=180)
    plt.close(fig)


def token_audit(root, output):
    counts, totals, cells, examples = [], [], [], []
    for path in sorted(root.glob('layer_*/summary.json')):
        layer = int(path.parent.name.split('_')[-1])
        for filename, task in [('trials.csv','presence')]:
            columns = ['kind','family','matching','dose','mapping','top_token_id','top_token_text',
                       'top_token_is_choice','response_letter','logit_x','logit_y','clean_token_norm','realized_amplitude']
            for chunk in pd.read_csv(path.parent / filename, usecols=columns, chunksize=100000, keep_default_na=False, low_memory=False):
                chunk = chunk[~chunk.kind.eq('sham')].copy()
                chunk['dose'] = pd.to_numeric(chunk['dose'], errors='raise')
                chunk['unrestricted_xy'] = chunk.top_token_is_choice.astype(str).str.lower().eq('true')
                chunk['restricted_tie'] = chunk.response_letter.eq('tie')
                invalid = chunk[~chunk.unrestricted_xy]
                totals.append(dict(layer=layer,task=task,n=len(chunk),unrestricted_xy=int(chunk.unrestricted_xy.sum()),ties=int(chunk.restricted_tie.sum())))
                tokens = invalid.groupby(['top_token_id','top_token_text'], dropna=False).size().reset_index(name='n')
                tokens['task'] = task
                counts.append(tokens)
                keys = ['family','matching','dose','mapping']
                for key, group in chunk.groupby(keys):
                    cells.append(dict(layer=layer,task=task,**dict(zip(keys,key)),n=len(group),
                                      unrestricted_xy=int(group.unrestricted_xy.sum()),ties=int(group.restricted_tie.sum())))
                if not invalid.empty:
                    example = invalid.head(3).copy()
                    example['layer'], example['task'] = layer, task
                    examples.append(example)
    counts = pd.concat(counts).groupby(['task','top_token_id','top_token_text'],dropna=False)['n'].sum().reset_index().sort_values(['task','n'],ascending=[True,False])
    counts.assign(top_token_text=counts.top_token_text.map(json.dumps)).rename(columns={'top_token_text':'top_token_text_json'}).to_csv(output / 'non_xy_top_tokens.csv',index=False)
    totals = pd.DataFrame(totals).groupby(['layer','task'],as_index=False)[['n','unrestricted_xy','ties']].sum()
    totals.to_csv(output / 'token_compliance_by_layer.csv',index=False)
    cells = pd.DataFrame(cells).groupby(['layer','task','family','matching','dose','mapping'],as_index=False)[['n','unrestricted_xy','ties']].sum()
    cells['unrestricted_xy_rate'] = cells.unrestricted_xy / cells.n
    cells.to_csv(output / 'token_compliance_by_cell.csv',index=False)
    if examples:
        example_table = pd.concat(examples)
        example_table.assign(top_token_text=example_table.top_token_text.map(json.dumps)).rename(columns={'top_token_text':'top_token_text_json'}).to_csv(output / 'non_xy_examples.csv',index=False)
    return counts, totals


def plot_score_edge_cases(root, output):
    cases = [(0, 'random'), (12, 'noise'), (20, 'concept')]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), layout='constrained')
    for col, (layer, family) in enumerate(cases):
        frame = pd.read_csv(root / f'layer_{layer}/trials.csv', usecols=['family','matching','dose','mapping','score','sham_score'])
        frame = frame[frame.family.eq(family) & frame.matching.eq('alpha') & frame.dose.eq(128)]
        for row, mapping in enumerate(('XY', 'YX')):
            ax = axes[row,col]
            group = frame[frame.mapping.eq(mapping)]
            if group.empty:
                continue
            values = np.r_[group.score, group.sham_score, 0.]
            bins = np.linspace(values.min() - .2, values.max() + .2, 25)
            ax.hist(group.sham_score,bins=bins,density=True,color='#64748b',alpha=.45,label='Matched clean score')
            ax.hist(group.score,bins=bins,density=True,color='#2563eb',alpha=.55,label='Perturbed score')
            ax.axvline(0,color='black',ls='--',label='Intervention decision threshold')
            ax.set(title=f'L{layer}, {family}, α = 128, {mapping}', xlabel='Recoded score: positive favors intervention', ylabel='Density')
            ax.grid(alpha=.15)
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside lower center', ncol=3, frameon=False)
    fig.suptitle('Three edge cases behind the hit-rate curves\nDescriptive distributions; matched shams carry the same weights as their intervention references',fontsize=14)
    fig.savefig(output / 'fig10_hit_rate_edge_cases.png',dpi=180)
    plt.close(fig)


def generate_diagnostics(root, calibration_csv, output):
    output.mkdir(parents=True, exist_ok=True)
    table = conversion_table(root, calibration_csv)
    table.to_csv(output / 'alpha_to_z.csv', index=False)
    plot_conversions(table, output)
    token_audit(root, output)
    plot_score_edge_cases(root, output)
    metadata = {
        'calibration_csv': str(calibration_csv.resolve()),
        'conversion': 'z_sd = alpha / sd; z_mad = alpha / mad_corrected',
        'bands': 'Range across selected directions; noise uses its full calibration bank',
        'dropout': 'Uses median fixed-random scale as in experiment; requested RMS amplitude, not realized norm',
        'calibration_uncertainty': 'Not propagated; plots condition on stored scale estimates',
        'response_audit': 'Unrestricted argmax at one next-token position, not generated continuations',
    }
    (output / 'diagnostics_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--calibration-scales',type=Path,default=Path(__file__).resolve().parents[2] / 'results/experiment_0_calibration/directional_scales.csv')
    args = parser.parse_args()
    generate_diagnostics(args.input_dir,args.calibration_scales,args.output_dir or args.input_dir / 'analysis')
