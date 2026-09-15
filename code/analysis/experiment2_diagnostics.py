"""Token-compliance diagnostics for the locked Experiment 2 run."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def token_audit(root, output):
    counts, totals, cells, examples = [], [], [], []
    for path in sorted(root.glob('layer_*/summary.json')):
        layer = int(path.parent.name.split('_')[-1])
        columns = [
            'kind', 'family', 'matching', 'dose', 'mapping', 'top_token_id',
            'top_token_text', 'top_token_is_choice', 'response_letter',
            'logit_x', 'logit_y', 'clean_token_norm', 'realized_amplitude',
        ]
        for chunk in pd.read_csv(
            path.parent / 'trials.csv',
            usecols=columns,
            chunksize=100000,
            keep_default_na=False,
            low_memory=False,
        ):
            chunk = chunk[~chunk.kind.eq('sham')].copy()
            chunk['dose'] = pd.to_numeric(chunk['dose'], errors='raise')
            chunk['unrestricted_xy'] = (
                chunk.top_token_is_choice.astype(str).str.lower().eq('true')
            )
            chunk['restricted_tie'] = chunk.response_letter.eq('tie')
            invalid = chunk[~chunk.unrestricted_xy]
            totals.append({
                'layer': layer,
                'task': 'presence',
                'n': len(chunk),
                'unrestricted_xy': int(chunk.unrestricted_xy.sum()),
                'ties': int(chunk.restricted_tie.sum()),
            })
            tokens = (
                invalid.groupby(['top_token_id', 'top_token_text'], dropna=False)
                .size()
                .reset_index(name='n')
            )
            tokens['task'] = 'presence'
            counts.append(tokens)
            keys = ['family', 'matching', 'dose', 'mapping']
            for key, group in chunk.groupby(keys):
                cells.append({
                    'layer': layer,
                    'task': 'presence',
                    **dict(zip(keys, key)),
                    'n': len(group),
                    'unrestricted_xy': int(group.unrestricted_xy.sum()),
                    'ties': int(group.restricted_tie.sum()),
                })
            if not invalid.empty:
                example = invalid.head(3).copy()
                example['layer'], example['task'] = layer, 'presence'
                examples.append(example)

    counts = (
        pd.concat(counts)
        .groupby(['task', 'top_token_id', 'top_token_text'], dropna=False)['n']
        .sum()
        .reset_index()
        .sort_values(['task', 'n'], ascending=[True, False])
    )
    counts.assign(
        top_token_text=counts.top_token_text.map(json.dumps)
    ).rename(
        columns={'top_token_text': 'top_token_text_json'}
    ).to_csv(output / 'non_xy_top_tokens.csv', index=False)

    totals = pd.DataFrame(totals).groupby(
        ['layer', 'task'], as_index=False
    )[['n', 'unrestricted_xy', 'ties']].sum()
    totals.to_csv(output / 'token_compliance_by_layer.csv', index=False)

    cells = pd.DataFrame(cells).groupby(
        ['layer', 'task', 'family', 'matching', 'dose', 'mapping'],
        as_index=False,
    )[['n', 'unrestricted_xy', 'ties']].sum()
    cells['unrestricted_xy_rate'] = cells.unrestricted_xy / cells.n
    cells.to_csv(output / 'token_compliance_by_cell.csv', index=False)

    if examples:
        example_table = pd.concat(examples)
        example_table.assign(
            top_token_text=example_table.top_token_text.map(json.dumps)
        ).rename(
            columns={'top_token_text': 'top_token_text_json'}
        ).to_csv(output / 'non_xy_examples.csv', index=False)
    return counts, totals


def generate_diagnostics(root, output):
    output.mkdir(parents=True, exist_ok=True)
    token_audit(root, output)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--output-dir', type=Path)
    args = parser.parse_args()
    generate_diagnostics(args.input_dir, args.output_dir or args.input_dir / 'analysis')
