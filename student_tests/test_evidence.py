"""Evidence contracts: disk tampering, pairing drift and unsafe baseline replacement."""
import copy
import importlib
import importlib.util
import importlib.metadata
import json
from types import SimpleNamespace

import pytest

from labkit import evaluate as ev
from labkit.config import get_tier


def evidence():
    assert importlib.util.find_spec('labkit.evidence') is not None, 'missing evidence implementation'
    return importlib.import_module('labkit.evidence')


@pytest.fixture
def experiment(tmp_path):
    root = tmp_path
    (root / 'data').mkdir()
    label = {'intent': 'doi_tra', 'urgency': 'cao', 'product': 'áo', 'sentiment': 'tieu_cuc'}
    target = [{'input': 'Ticket A', 'label': label}, {'input': 'Ticket B', 'label': label},
              {'input': 'Ticket C', 'label': label}]
    regression = [{'instruction': 'Instruction A', 'keywords': ['alpha']},
                  {'instruction': 'Instruction B', 'keywords': ['beta']}]
    for name, rows in [('eval_target', target), ('eval_regression', regression)]:
        (root / 'data' / f'{name}.jsonl').write_text('\n'.join(json.dumps(r, ensure_ascii=False) for r in rows), encoding='utf-8')
    (root / 'data' / 'train_seed.jsonl').write_text('{}\n', encoding='utf-8')
    good = json.dumps(label, ensure_ascii=False)
    a, b = ['bad', 'bad', 'bad'], [good, 'bad', good]
    ra, rb = ['bad', 'bad'], ['alpha', 'beta']
    frozen = {'tier': 'T4', 'model': get_tier('T4').model_id,
              'n_target': 3, 'n_regression': 2, 'eval_limit': None, 'smoke_mode': False,
              'baseline_a': ev.GroupScores(target=0, regression=0, format=0, n=3).as_dict(),
              'baseline_b': ev.GroupScores(target=2/3, regression=1, format=2/3, n=3).as_dict()}
    return SimpleNamespace(root=root, target=target, regression=regression, a=a, b=b, ra=ra, rb=rb,
                           frozen=frozen, good=good)


def context(x):
    return evidence().experiment_context(x.root, get_tier('T4'), precision='fp16')


def save(x, **kwargs):
    ctx = context(x)
    frozen = copy.deepcopy(x.frozen)
    frozen['optimized_prompt_sha'] = ctx['optimized_prompt_sha'][:16]
    evidence().record_baseline(x.root, ctx, x.target, x.regression, x.a, x.b, x.ra, x.rb,
                               frozen, **kwargs)
    return ctx


def test_paired_scores_use_raw_outputs_and_preserve_order(experiment):
    x = experiment
    ctx = save(x)
    raw = evidence().verify_baseline(x.root, ctx, x.target, x.regression)
    paired = evidence().pair_qualitative(x.target, raw['target'], ['bad', x.good, x.good])
    assert [r['i'] for r in paired] == [0, 1, 2]
    assert [r['ticket'] for r in paired] == ['Ticket A', 'Ticket B', 'Ticket C']
    assert [r['delta'] for r in paired] == [-1.0, 1.0, 0.0]
    assert paired[0]['baseline_b_pred'] == x.good
    assert paired[0]['label'] == x.target[0]['label']


@pytest.mark.parametrize('change', ['length', 'order', 'input', 'label', 'index'])
def test_pairing_rejects_silent_zip_truncation_or_identity_drift(experiment, change):
    x = experiment
    ctx = save(x)
    raw = evidence().verify_baseline(x.root, ctx, x.target, x.regression)['target']
    if change == 'length':
        raw.pop()
    elif change == 'order':
        raw.reverse()
    elif change == 'input':
        raw[0]['ticket'] = 'Other ticket'
    elif change == 'label':
        raw[0]['label']['urgency'] = 'thap'
    else:
        raw[0]['i'] = 7
    with pytest.raises(ValueError):
        evidence().pair_qualitative(x.target, raw, x.a)


def test_pairing_rejects_short_ft_outputs(experiment):
    x = experiment
    ctx = save(x)
    raw = evidence().verify_baseline(x.root, ctx, x.target, x.regression)
    with pytest.raises(ValueError):
        evidence().pair_qualitative(x.target, raw['target'], ['bad'])


@pytest.mark.parametrize('filename', ['baseline_predictions.json', 'baselines_frozen.json', 'environment.txt'])
def test_corrupted_disk_artifacts_fail_integrity_before_scoring(experiment, filename):
    x = experiment
    ctx = save(x)
    path = x.root / 'results' / filename
    path.write_text(path.read_text(encoding='utf-8') + ' ', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        evidence().verify_baseline(x.root, ctx, x.target, x.regression)


@pytest.mark.parametrize('field,value', [('model', 'other/model'), ('tier', 'BIGGPU'),
                                        ('optimized_prompt_sha', 'different'), ('naive_prompt_sha', 'different'),
                                        ('epochs', 1), ('max_length', 512), ('seed', 7),
                                        ('precision', 'bf16'), ('eval_limit', 1)])
def test_frozen_context_mismatch_is_rejected(experiment, field, value):
    x = experiment
    ctx = save(x)
    ctx[field] = value
    with pytest.raises(ValueError, match='context'):
        evidence().verify_baseline(x.root, ctx, x.target, x.regression)


def test_same_count_eval_edit_is_rejected(experiment):
    x = experiment
    save(x)
    path = x.root / 'data' / 'eval_regression.jsonl'
    path.write_text(path.read_text(encoding='utf-8').replace('alpha', 'changed'), encoding='utf-8')
    with pytest.raises(ValueError, match='context'):
        evidence().verify_baseline(x.root, context(x), x.target, x.regression)


@pytest.mark.parametrize('kind', ['target_order', 'target_label', 'regression_order', 'regression_label'])
def test_in_memory_eval_identity_must_match_saved_rows(experiment, kind):
    x = experiment
    ctx = save(x)
    target, regression = copy.deepcopy(x.target), copy.deepcopy(x.regression)
    if kind == 'target_order':
        target.reverse()
    elif kind == 'target_label':
        target[0]['label']['urgency'] = 'thap'
    elif kind == 'regression_order':
        regression.reverse()
    else:
        regression[0]['keywords'] = ['changed']
    with pytest.raises(ValueError):
        evidence().verify_baseline(x.root, ctx, target, regression)


def test_existing_baseline_needs_explicit_pretraining_rerun(experiment):
    x = experiment
    save(x)
    original = (x.root / 'results' / 'experiment_manifest.json').read_bytes()
    with pytest.raises(ValueError, match='rerun'):
        save(x)
    assert (x.root / 'results' / 'experiment_manifest.json').read_bytes() == original
    save(x, replace_before_training=True)
    archives = list((x.root / 'results' / 'baseline_history').glob('*/experiment_manifest.json'))
    assert len(archives) == 1
    assert archives[0].read_bytes() == original


@pytest.mark.parametrize('before_first_record', [False, True])
def test_adapter_blocks_baseline_record_even_with_rerun_flag(experiment, before_first_record):
    x = experiment
    if not before_first_record:
        save(x)
    adapter = x.root / 'adapters' / 'correct'
    adapter.mkdir(parents=True)
    (adapter / 'adapter_model.safetensors').write_bytes(b'trained')
    with pytest.raises(ValueError, match='training'):
        save(x, replace_before_training=True)


def test_inconsistent_existing_bundle_cannot_be_replaced(experiment):
    x = experiment
    save(x)
    path = x.root / 'results' / 'baseline_predictions.json'
    path.write_text('[]', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        save(x, replace_before_training=True)
    assert path.read_text(encoding='utf-8') == '[]'


def test_record_rejects_short_regression_predictions_without_writing(experiment):
    x = experiment
    x.ra.pop()
    with pytest.raises(ValueError, match='length'):
        save(x)
    assert not (x.root / 'results' / 'experiment_manifest.json').exists()


def test_frozen_aggregate_scores_must_match_raw_predictions(experiment):
    x = experiment
    x.frozen['baseline_b']['target'] = 1.0
    with pytest.raises(ValueError, match='score'):
        save(x)


def test_manifest_records_actual_versions_and_raw_regression_on_disk(experiment):
    x = experiment
    ctx = save(x)
    manifest = json.loads((x.root / 'results' / 'experiment_manifest.json').read_text(encoding='utf-8'))
    assert manifest['context'] == ctx
    assert manifest['created_at_utc'].endswith('+00:00')
    assert 'python' in manifest['environment']
    assert manifest['environment']['pytest'] == importlib.metadata.version('pytest')
    assert 'source_commit' in manifest
    assert manifest['model_revision'] is None
    raw = json.loads((x.root / 'results' / 'baseline_predictions.json').read_text(encoding='utf-8'))
    assert raw['regression'][0]['instruction'] == 'Instruction A'
    assert raw['regression'][0]['keywords'] == ['alpha']
    assert raw['regression'][0]['baseline_a_pred'] == 'bad'
    assert raw['regression'][0]['baseline_b_pred'] == 'alpha'
    assert raw['target'][0]['ticket'] == 'Ticket A'


def test_known_resolved_revision_changes_are_rejected(experiment):
    x = experiment
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash='model123'))
    tokenizer = SimpleNamespace(name_or_path='tokenizer/id', init_kwargs={'_commit_hash': 'token123'})
    ctx = save(x, model=model, tokenizer=tokenizer)
    evidence().verify_model_revisions(x.root, model, tokenizer)
    model.config._commit_hash = 'different'
    with pytest.raises(ValueError, match='revision'):
        evidence().verify_model_revisions(x.root, model, tokenizer)
    model.config._commit_hash = None
    with pytest.raises(ValueError, match='revision'):
        evidence().verify_model_revisions(x.root, model, tokenizer)


def test_zero_losses_are_reported_without_manufacturing_examples(experiment):
    x = experiment
    ctx = save(x)
    raw = evidence().verify_baseline(x.root, ctx, x.target, x.regression)
    pairs = evidence().pair_qualitative(x.target, raw['target'], [x.good] * 3)
    summary = evidence().summarize_pairs(pairs)
    assert summary['losses'] == 0
    assert summary['wins'] == 1
    assert summary['ties'] == 2
    assert summary['two_loss_examples_available'] is False
    assert len(summary['selected_examples']) == 3
    assert all(row['delta'] >= 0 for row in summary['selected_examples'])

# Execute the actual notebook scripts, replacing only the external GPU/model boundary.
@pytest.fixture
def notebook_runtime(experiment, monkeypatch):
    import sys
    from pathlib import Path
    from labkit import device, generate
    x = experiment
    scripts = Path(__file__).resolve().parents[1] / 'notebooks'
    monkeypatch.chdir(x.root)
    monkeypatch.setenv('COMPUTE_TIER', 'T4')
    monkeypatch.setenv('EPOCHS', '2')
    for name in ('BASE_MODEL', 'EVAL_LIMIT', 'BASELINE_RERUN'):
        monkeypatch.delenv(name, raising=False)
    calls = {'load': 0, 'generate': 0}
    model = SimpleNamespace(config=SimpleNamespace(_commit_hash='model123'), eval=lambda: None)
    tok = SimpleNamespace(name_or_path=get_tier('T4').model_id, init_kwargs={'_commit_hash': 'token123'})

    def load_base(tier, **kwargs):
        calls['load'] += 1
        return model, tok

    def generate_batch(model, tokenizer, prompts, *, label, **kwargs):
        calls['generate'] += 1
        if label.startswith('(a)'):
            return (x.a if label.endswith('/target') else x.ra), 10.0
        if label.startswith('(b)'):
            return (x.b if label.endswith('/target') else x.rb), 10.0
        return (['bad', x.good, x.good] if label.endswith('/target') else x.rb), 10.0

    monkeypatch.setattr(device, 'precision', lambda: 'fp16')
    monkeypatch.setattr(generate, 'load_base', load_base)
    monkeypatch.setattr(generate, 'generate_batch', generate_batch)
    monkeypatch.setattr(generate, 'free_memory', lambda: None)
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(
        PeftModel=SimpleNamespace(from_pretrained=lambda model, path: model)))
    return x, scripts, calls


def test_notebooks_reuse_generated_outputs_and_emit_true_paired_examples(notebook_runtime):
    import runpy
    x, scripts, calls = notebook_runtime
    runpy.run_path(str(scripts / '02_baselines.py'))
    assert (x.root / 'results' / 'experiment_manifest.json').exists(), 'NB2 did not capture provenance'
    assert calls['generate'] == 4  # a/b x target/regression; evidence must add zero sweeps.
    runpy.run_path(str(scripts / '05_evaluate_and_verdict.py'))
    assert calls['generate'] == 6  # NB5 correct target/regression only; no contrast adapters exist.
    pairs = json.loads((x.root / 'results' / 'paired_qualitative.json').read_text(encoding='utf-8'))
    assert [r['delta'] for r in pairs] == [-1.0, 1.0, 0.0]
    summary = json.loads((x.root / 'results' / 'paired_qualitative_summary.json').read_text(encoding='utf-8'))
    assert summary['losses'] == 1
    assert summary['two_loss_examples_available'] is False
    assert (x.root / 'results' / 'qualitative.json').exists()
    assert (x.root / 'results' / 'verdict.json').exists()


def test_nb5_rejects_corruption_before_any_model_or_gpu_call(notebook_runtime):
    import runpy
    x, scripts, calls = notebook_runtime
    save(x)
    path = x.root / 'results' / 'baseline_predictions.json'
    path.write_text('corrupted', encoding='utf-8')
    with pytest.raises(ValueError, match='hash'):
        runpy.run_path(str(scripts / '05_evaluate_and_verdict.py'))
    assert calls == {'load': 0, 'generate': 0}


def test_nb2_rejects_posttraining_rerun_before_any_model_or_gpu_call(notebook_runtime):
    import runpy
    x, scripts, calls = notebook_runtime
    adapter = x.root / 'adapters' / 'correct'
    adapter.mkdir(parents=True)
    (adapter / 'adapter_model.safetensors').write_bytes(b'trained')
    with pytest.raises(ValueError, match='training'):
        runpy.run_path(str(scripts / '02_baselines.py'))
    assert calls == {'load': 0, 'generate': 0}
