#!/usr/bin/env python3
"""Export a compact detection-only snapshot from an Experiment 2 run.

Raw trials stay outside Git. SHA-256 hashes identify inputs; only detection cells
are copied from historical combined summaries. No semantic-control file is read.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

from experiment2_controls import generate_controls
from experiment2_diagnostics import token_audit


def layer_root(root):
    candidate = root / 'layers'
    return candidate if candidate.is_dir() else root


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def export(root, calibration, output):
    output.mkdir(parents=True, exist_ok=True)
    cells, provenance = [], []
    for path in sorted(layer_root(root).glob('layer_*/summary.json')):
        source = json.loads(path.read_text())
        if source.get('experiment') != 2:
            raise ValueError(f'Expected Experiment 2: {path}')
        cells.extend(source['cells'])
        directory = path.parent
        manifest = json.loads((directory / 'manifest.json').read_text())
        provenance.append({
            'detection_design': {key: manifest[key] for key in [
                'model','model_revision','tokenizer_revision','seed','pairs','concepts',
                'dose_grid','label_orders','mappings','response_protocol','answer_token_ids',
                'implementation_sha256','dropout_norm_source',
            ] if key in manifest},
            'layer_directory': directory.name,
            'source_protocol_id': source.get('protocol_id'),
            'bootstrap_iterations': source.get('bootstrap_iterations'),
            'input_sha256': {name: digest(directory / name) for name in
                             ['summary.json', 'manifest.json', 'trials.csv']},
            'calibration_provenance': source.get('calibration_provenance'),
        })
    if not cells:
        raise ValueError('No detection cells found')
    with gzip.open(output / 'snapshot.json.gz', 'wt') as stream:
        json.dump({'experiment':2,'cells':cells}, stream, sort_keys=True)
    generate_controls(root, output)
    token_audit(root, output)
    (output / 'provenance.json').write_text(json.dumps({
        'experiment':2, 'source_run_name':root.name,
        'extraction':'Only summary.cells and trials.csv; original protocol IDs retained below. These are historical results, not a new v3 inference run.',
        'calibration_sha256':digest(calibration), 'layers':provenance,
        'bootstrap_note':'Stored crossed-bootstrap intervals are copied unchanged. Clean-control intervals are recomputed with 2000 pair-cluster draws, seed 20260914.',
    }, indent=2)+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input_dir', type=Path)
    parser.add_argument('--calibration-scales', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    export(args.input_dir, args.calibration_scales, args.output_dir)
