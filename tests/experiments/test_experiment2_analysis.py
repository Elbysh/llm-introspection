"""Statistical safeguards for the clean-control analysis."""
import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

spec = importlib.util.spec_from_file_location(
    'experiment2_controls', Path(__file__).resolve().parents[2] / 'code/analysis/experiment2_controls.py'
)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def test_sham_references_are_not_independent_observations():
    frame = pd.DataFrame({'sham_trial_id': ['a', 'a', 'b'], 'score': [1., 1., -1.]})
    unique = analysis.unique_shams(frame, ['score'])
    assert len(unique) == 2
    assert unique.score.mean() == 0
    frame.loc[1, 'score'] = 2
    with pytest.raises(ValueError, match='Conflicting'):
        analysis.unique_shams(frame, ['score'])


def test_cluster_bootstrap_keeps_complementary_mapping_pairs_together():
    frame = pd.DataFrame({'prompt': [0, 0, 1, 1], 'decision': [0., 1., 0., 1.]})
    point, ci, draws = analysis.cluster_estimates(
        frame, ['decision'], 'prompt', 100, np.random.default_rng(7)
    )
    assert point['decision'] == .5
    assert np.all(ci == .5)
    assert np.all(draws == .5)


def test_cluster_bootstrap_is_reproducible_and_non_degenerate():
    frame = pd.DataFrame({'prompt': [0, 0, 1, 1], 'score': [-1., -1., 1., 1.]})
    first = analysis.cluster_estimates(frame, ['score'], 'prompt', 100, np.random.default_rng(7))
    second = analysis.cluster_estimates(frame, ['score'], 'prompt', 100, np.random.default_rng(7))
    np.testing.assert_array_equal(first[2], second[2])
    assert first[1][0, 0] < first[0]['score'] < first[1][1, 0]


diag_spec = importlib.util.spec_from_file_location(
    'experiment2_diagnostics', Path(__file__).resolve().parents[2] / 'code/analysis/experiment2_diagnostics.py'
)
diag = importlib.util.module_from_spec(diag_spec)
diag_spec.loader.exec_module(diag)


def test_alpha_to_z_conversion_and_invalid_scales():
    np.testing.assert_allclose(diag.alpha_to_z(128, [2, 4]), [64, 32])
    for scale in [0, -1, np.nan, np.inf]:
        with pytest.raises(ValueError, match='finite and positive'):
            diag.alpha_to_z(128, scale)


def test_conversion_uses_selected_directions_and_dropout_bank_median(tmp_path):
    import json
    layer = tmp_path / 'layer_0'
    layer.mkdir()
    plan = [dict(family=f, direction_id=d, matching='alpha', dose=128)
            for f, d in [('concept', 'c'), ('random', 'r1'), ('noise', 'n'), ('dropout', 'd')]]
    (layer / 'manifest.json').write_text(json.dumps({'plan': plan}))
    scales = pd.DataFrame({
        'direction_id': ['c', 'r1', 'r2', 'n'], 'decoder_block_index': [0]*4,
        'direction_family': ['concept', 'fixed_random', 'fixed_random', 'renewed_noise'],
        'sd': [2., 4., 8., 16.], 'mad_corrected': [1., 2., 4., 8.],
    })
    path = tmp_path / 'scales.csv'
    scales.to_csv(path, index=False)
    table = diag.conversion_table(tmp_path, path)
    sd = table[table.estimator.eq('sd')].set_index('family')
    assert sd.loc['random', 'z'] == 32
    assert sd.loc['dropout', 'z'] == 128 / 6
    assert sd.loc['noise', 'z'] == 8
    assert table[table.family.eq('random')].direction_id.nunique() == 1


def test_export_excludes_other_task_even_from_combined_summary(tmp_path, monkeypatch):
    import gzip
    import json
    folder = Path(__file__).resolve().parents[2] / 'code/analysis'
    monkeypatch.syspath_prepend(str(folder))
    import export_experiment2 as exporter
    root = tmp_path / 'source'
    layer = root / 'layer_1'
    layer.mkdir(parents=True)
    (layer / 'summary.json').write_text(json.dumps({
        'experiment': 2, 'cells': [{'layer': 1, 'auroc': .75}],
        'capability_cells': [{'accuracy': .99}], 'protocol_id':'historical-combined',
    }))
    (layer / 'manifest.json').write_text(json.dumps({'model':'test', 'capability_control':{'secret':'exclude'}}))
    (layer / 'trials.csv').write_text('detection-only\n')
    calibration = tmp_path / 'calibration.csv'
    calibration.write_text('placeholder')
    monkeypatch.setattr(exporter, 'generate_controls', lambda *args: None)
    monkeypatch.setattr(exporter, 'token_audit', lambda *args: None)
    monkeypatch.setattr(exporter, 'conversion_table', lambda *args: pd.DataFrame([
        dict(layer=1, family='concept', estimator='sd', alpha=1, z=2)]))
    output = tmp_path / 'output'
    exporter.export(root, calibration, output)
    with gzip.open(output / 'snapshot.json.gz', 'rt') as stream:
        snapshot = json.load(stream)
    assert snapshot == {'experiment':2, 'cells':[{'layer':1,'auroc':.75}]}
    provenance = json.loads((output / 'provenance.json').read_text())
    assert provenance['layers'][0]['detection_design'] == {'model':'test'}
    assert 'capability' not in json.dumps(provenance)
    assert provenance['layers'][0]['input_sha256']['trials.csv'] == exporter.digest(layer / 'trials.csv')
