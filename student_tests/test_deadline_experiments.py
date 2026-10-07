"""CPU contract tests for supplementary GPU stages; no torch import required."""
import importlib.util
import json
import csv
from datetime import datetime, timezone, timedelta
from pathlib import Path
from types import SimpleNamespace
import pytest
from labkit import evidence, evaluate as ev
from labkit.config import get_tier

SCRIPT = Path(__file__).parents[1] / 'student_tools' / 'deadline_experiments.py'


def helper():
    assert SCRIPT.exists(), 'GPU follow-up runner is missing'
    spec = importlib.util.spec_from_file_location('deadline_experiments', SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def experiment(tmp_path, monkeypatch):
    monkeypatch.setenv('EPOCHS', '2')
    monkeypatch.delenv('BASE_MODEL', raising=False)
    root = tmp_path
    (root / 'data/split').mkdir(parents=True)
    label = {'intent': 'doi_tra', 'urgency': 'cao', 'product': 'ao', 'sentiment': 'tieu_cuc'}
    target = [{'input': f'Ticket {i}', 'label': label} for i in range(50)]
    regression = [{'instruction': f'Question {i}', 'keywords': ['alpha']} for i in range(15)]
    for name, rows in [('eval_target', target), ('eval_regression', regression), ('train_seed', [{}] * 250), ('split/train', [{}] * 225), ('split/val', [{}] * 25)]:
        (root / 'data' / f'{name}.jsonl').write_text('\n'.join(json.dumps(r) for r in rows), encoding='utf-8')
    ctx = evidence.experiment_context(root, get_tier('T4'), precision='fp16')
    good = json.dumps(label)
    frozen = {'tier': 'T4', 'model': get_tier('T4').model_id, 'n_target': 50, 'n_regression': 15, 'eval_limit': None, 'smoke_mode': False, 'optimized_prompt_sha': ctx['optimized_prompt_sha'][:16], 'baseline_a': ev.GroupScores(target=0, regression=0, format=0, n=50).as_dict(), 'baseline_b': ev.GroupScores(target=1, regression=1, format=1, n=50).as_dict()}
    evidence.record_baseline(root, ctx, target, regression, ['bad'] * 50, [good] * 50, ['bad'] * 15, ['alpha'] * 15, frozen)
    (root / 'results/verdict.json').write_text(json.dumps({'comparison': [{'run': '(c) LoRA fine-tune', 'regression': .5222}], 'verdict': {'passed': False}}), encoding='utf-8')
    with (root / 'results/runs.csv').open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['run', 'model', 'tier', 'precision', 'load_in_4bit', 'r', 'lora_alpha', 'learning_rate', 'placement', 'mask_mode', 'max_steps', 'train_seconds', 'peak_vram_gb', 'final_loss', 'trainable_params'])
        writer.writeheader()
        for key, rank in [('correct', 16), ('attn_only', 283)]:
            writer.writerow(dict(run=key, model=ctx['model'], tier='T4', precision='fp16', load_in_4bit=False, r=rank, lora_alpha=rank*2, learning_rate=.0001, placement='text-linear' if key == 'correct' else 'attn-only', mask_mode='assistant-only', max_steps=30, train_seconds=416, peak_vram_gb=8.78, final_loss=.62, trainable_params=123))
            folder = root / 'adapters' / key
            folder.mkdir(parents=True)
            (folder / 'adapter_config.json').write_text(json.dumps({'base_model_name_or_path': ctx['model'], 'r': rank, 'lora_alpha': 2*rank}), encoding='utf-8')
            (folder / 'adapter_model.safetensors').write_bytes(b'actual mock adapter')
    return SimpleNamespace(root=root, good=good)


class FakeBackend:
    def __init__(self, good):
        self.good, self.calls = good, []
        self.tokenizer = SimpleNamespace(init_kwargs={}, name_or_path=None)
    def load_adapter(self, path):
        self.calls.append(('load', path.name))
        return SimpleNamespace(config=SimpleNamespace(_commit_hash=None)), self.tokenizer
    def generate(self, model, tok, prompts, **kwargs):
        self.calls.append(('generate', len(prompts), kwargs))
        return [('alpha' if p.startswith('Question') else self.good) for p in prompts], 1.0
    def merge(self, model):
        self.calls.append(('merge',))
        return model
    def load_hot_swap(self, paths):
        self.calls.append(('hot_load', tuple(paths)))
        model = SimpleNamespace(config=SimpleNamespace(_commit_hash=None), identity='same-base-1')
        return model, self.tokenizer, 'same-base-1'
    def select_adapter(self, model, key):
        self.calls.append(('select', key, model.identity))
    def train_rank(self, root, rank, train_rows):
        self.calls.append(('train', rank, len(train_rows)))
        model = SimpleNamespace(config=SimpleNamespace(_commit_hash=None))
        return model, self.tokenizer, {'r': rank, 'lora_alpha': rank*2, 'learning_rate': .0001, 'max_steps': 30, 'placement': 'text-linear', 'train_seconds': 1, 'peak_vram_gb': 1, 'final_loss': .5}
    def release(self):
        self.calls.append(('release',))


def make_runner(experiment, **kwargs):
    mod = helper()
    backend = FakeBackend(experiment.good)
    now = datetime(2026, 10, 7, 15, 0, tzinfo=timezone.utc)
    runner = mod.Runner(experiment.root, backend=backend, now=lambda: now, stop_at=now + timedelta(hours=2), **kwargs)
    return runner, backend


def test_regression_generates_exactly_fifteen_with_raw_pairing(experiment):
    runner, backend = make_runner(experiment)
    status = runner.run(['regression'])
    raw = json.loads((experiment.root / 'bonus_results/regression_followup.json').read_text())
    assert status['stages']['regression']['status'] == 'complete'
    generations = [c for c in backend.calls if c[0] == 'generate']
    assert len(generations) == 1 and generations[0][1] == 15
    assert generations[0][2]['max_new_tokens'] == 96
    assert generations[0][2]['system'] is None and generations[0][2]['enable_thinking'] is False
    assert len(raw['rows']) == 15
    assert raw['rows'][0]['baseline_b_pred'] == 'alpha'
    assert raw['rows'][0]['delta'] == 0
    assert raw['supplementary_repeat'] is True
    assert raw['original_ft_regression_aggregate'] == .5222
    assert 'unavailable' in raw['metadata']['revision_caution'].lower()


@pytest.mark.parametrize('bad', ['baseline', 'adapter'])
def test_preflight_failure_happens_before_any_gpu_load(experiment, bad):
    runner, backend = make_runner(experiment)
    if bad == 'baseline':
        path = experiment.root / 'results/baseline_predictions.json'
        path.write_text(path.read_text() + ' ')
    else:
        path = experiment.root / 'adapters/correct/adapter_config.json'
        path.write_text('{}')
    with pytest.raises(ValueError):
        runner.run(['regression'])
    assert not backend.calls


def test_b1_really_merges_fifty_and_hot_swaps_two_on_same_base(experiment):
    runner, backend = make_runner(experiment)
    runner.run(['b1'])
    merge = json.loads((experiment.root / 'bonus_results/merge_check.json').read_text())
    swap = json.loads((experiment.root / 'bonus_results/hot_swap.json').read_text())
    assert ('merge',) in backend.calls
    assert len(merge['before']['rows']) == len(merge['after']['rows']) == 50
    assert merge['tolerance'] == .01 and merge['passed']
    selects = [c for c in backend.calls if c[0] == 'select']
    assert [c[1] for c in selects] == ['correct', 'attn_only']
    assert len({c[2] for c in selects}) == 1
    assert set(swap['adapters']) == {'correct', 'attn_only'}
    assert all(len(x['rows']) == 50 for x in swap['adapters'].values())
    assert [c[1] for c in backend.calls if c[0] == 'generate'] == [50, 50, 50, 50]


def test_b4_trains_only_eight_and_sixtyfour_reuses_sixteen(experiment):
    runner, backend = make_runner(experiment)
    runner.run(['b4'])
    raw = json.loads((experiment.root / 'bonus_results/rank_sweep.json').read_text())
    assert raw['status'] == 'complete'
    assert [c[1] for c in backend.calls if c[0] == 'train'] == [8, 64]
    assert [c[1] for c in backend.calls if c[0] == 'generate'] == [50, 50, 50]
    assert raw['ranks']['16']['reused_original_adapter'] is True
    assert all(len(x['rows']) == 50 for x in raw['ranks'].values())
    assert (experiment.root / 'bonus_results/bonus_runs.csv').exists()


def test_expired_deadline_and_insufficient_stage_budget_load_nothing(experiment):
    runner, backend = make_runner(experiment)
    runner.stop_at = runner.now() - timedelta(seconds=1)
    status = runner.run(['regression', 'b1', 'b4'])
    assert all(v['status'] == 'incomplete' for v in status['stages'].values())
    assert not backend.calls


def test_partial_rank_sweep_persists_honest_status(experiment):
    runner, backend = make_runner(experiment)
    runner.stop_at = runner.now() + timedelta(seconds=200)
    runner.run(['b4'])
    raw = json.loads((experiment.root / 'bonus_results/rank_sweep.json').read_text())
    assert raw['status'] == 'partial' and set(raw['ranks']) == {'16'}
    assert not [c for c in backend.calls if c[0] == 'train']


def test_seal_detects_mutation_even_when_generation_raises(experiment):
    runner, backend = make_runner(experiment)
    path = experiment.root / 'results/verdict.json'
    def corrupt(*args, **kwargs):
        path.write_text('{}')
        raise RuntimeError('GPU OOM')
    backend.generate = corrupt
    with pytest.raises(ValueError, match='sealed'):
        runner.run(['regression'])
    status = json.loads((experiment.root / 'bonus_results/run_status.json').read_text())
    assert status['seal_unchanged'] is False


def test_success_preserves_all_original_bytes(experiment):
    before = {str(p.relative_to(experiment.root)): p.read_bytes() for folder in ['results', 'data', 'adapters'] for p in (experiment.root / folder).rglob('*') if p.is_file()}
    runner, backend = make_runner(experiment)
    runner.run(['regression', 'b1', 'b4'])
    assert before == {name: (experiment.root / name).read_bytes() for name in before}


def test_short_predictions_fail_without_claiming_completion(experiment):
    runner, backend = make_runner(experiment)
    backend.generate = lambda *args, **kwargs: (['alpha'], 1.0)
    with pytest.raises(ValueError, match='prediction length'):
        runner.run(['regression'])
    status = json.loads((experiment.root / 'bonus_results/run_status.json').read_text())
    assert status['stages']['regression']['status'] == 'failed'
    assert status['seal_unchanged'] is True
    assert not (experiment.root / 'bonus_results/regression_followup.json').exists()


def test_b1_insufficient_full_merge_budget_does_not_load(experiment):
    runner, backend = make_runner(experiment)
    runner.stop_at = runner.now() + timedelta(seconds=200)
    status = runner.run(['b1'])
    assert status['stages']['b1']['status'] == 'incomplete'
    assert not [c for c in backend.calls if c[0] == 'load']


def test_wrong_full_eval_count_fails_before_gpu(experiment):
    runner, backend = make_runner(experiment)
    path = experiment.root / 'data/eval_target.jsonl'
    path.write_text('\n'.join(path.read_text().splitlines()[:49]))
    with pytest.raises(ValueError, match='50/15'):
        runner.run(['b1'])
    assert not backend.calls


def test_available_revision_mismatch_stops_before_generation(experiment, monkeypatch):
    runner, backend = make_runner(experiment)
    def reject(*args):
        raise ValueError('model revision mismatch')
    monkeypatch.setattr(evidence, 'verify_model_revisions', reject)
    with pytest.raises(ValueError, match='revision mismatch'):
        runner.run(['regression'])
    assert not [c for c in backend.calls if c[0] == 'generate']


def test_real_backend_hot_swap_loads_two_named_adapters_and_one_base(experiment, monkeypatch):
    import sys
    mod = helper()
    calls = []
    base = object()
    class Peft:
        def __init__(self):
            self.active_adapter = 'correct'
        @classmethod
        def from_pretrained(cls, received, path, adapter_name):
            assert received is base
            calls.append(('from', adapter_name, path))
            return cls()
        def load_adapter(self, path, adapter_name):
            calls.append(('load', adapter_name, path))
        def get_base_model(self):
            return base
        def set_adapter(self, key):
            calls.append(('activate', key))
            self.active_adapter = key
        def eval(self):
            return self
    monkeypatch.setitem(sys.modules, 'peft', SimpleNamespace(PeftModel=Peft))
    backend = mod.GPUBackend(experiment.root, get_tier('T4'))
    loads = []
    def loadbase():
        loads.append('base')
        return base, object()
    backend._base = loadbase
    model, tok, identity = backend.load_hot_swap({'correct': experiment.root / 'adapters/correct', 'attn_only': experiment.root / 'adapters/attn_only'})
    for key in ['correct', 'attn_only']:
        backend.select_adapter(model, key)
    assert loads == ['base']
    assert [c[:2] for c in calls] == [('from', 'correct'), ('load', 'attn_only'), ('activate', 'correct'), ('activate', 'attn_only')]
    assert identity


def test_real_merge_invokes_safe_peft_merge(experiment):
    mod = helper()
    backend = mod.GPUBackend(experiment.root, get_tier('T4'))
    calls = []
    merged = SimpleNamespace(eval=lambda: calls.append('eval'))
    model = SimpleNamespace(merge_and_unload=lambda **kw: (calls.append(kw) or merged))
    assert backend.merge(model) is merged
    assert calls == [{'safe_merge': True}, 'eval']


def test_merge_improvement_above_one_percent_passes_non_drop_gate(experiment):
    runner, backend = make_runner(experiment)
    calls = []
    def predict(model, tok, prompts, **kwargs):
        calls.append(kwargs['label'])
        preds = [experiment.good] * len(prompts)
        if kwargs['label'] == 'correct/before-merge':
            preds[:5] = ['bad'] * 5
        return preds, 1.0
    backend.generate = predict
    runner.run(['b1'])
    raw = json.loads((experiment.root / 'bonus_results/merge_check.json').read_text())
    assert raw['target_delta'] > .01 and raw['passed'] is True


def test_rank_numeric_nan_log_retains_explicit_provenance_in_strict_json(experiment):
    runner, backend = make_runner(experiment)
    original_train = backend.train_rank
    logs = [{'step': 5, 'loss': .7, 'grad_norm': float('nan')}, {'step': 10, 'grad_norm': float('inf')}, {'step': 15, 'grad_norm': float('-inf')}]
    def train_with_nonfinite(root, rank, rows):
        model, tok, metrics = original_train(root, rank, rows)
        metrics['training_log_history'] = logs
        return model, tok, metrics
    backend.train_rank = train_with_nonfinite
    runner.run(['b4'])
    raw_text = (experiment.root / 'bonus_results/rank_sweep.json').read_text()
    raw = json.loads(raw_text, parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))
    assert raw['status'] == 'complete'
    assert raw['ranks']['8']['training_log_history'][0]['grad_norm'] == 'NaN'
    assert raw['ranks']['8']['effective_optimizer_updates'] is None
    assert 'unmeasured' in raw['ranks']['8']['effective_optimizer_updates_caution'].lower()
    assert raw['nonfinite_serialization']['count'] == 6
    assert any(entry['path'] == '$.ranks.8.training_log_history[0].grad_norm' and entry['value'] == 'NaN' for entry in raw['nonfinite_serialization']['entries'])
    assert raw['nonfinite_serialization']['counts'] == {'NaN': 2, '+Infinity': 2, '-Infinity': 2}
    assert logs[0]['grad_norm'] != logs[0]['grad_norm']  # Source logs retain original numeric NaN.


@pytest.mark.parametrize('stage,existing', [('regression', 'bonus_results/regression_followup.json'), ('b1', 'bonus_results/merge_check.json'), ('b1', 'bonus_results/hot_swap.json'), ('b4', 'bonus_results/rank_sweep.json'), ('b4', 'bonus_results/bonus_runs.csv'), ('b4', 'bonus_adapters/rank_8/adapter_model.safetensors'), ('b4', 'bonus_adapters/rank_64/adapter_model.safetensors')])
def test_existing_stage_evidence_refuses_rerun_without_loading_or_writing(experiment, stage, existing):
    path = experiment.root / existing
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b'prior attempt evidence, preserve exact bytes')
    before = {str(p.relative_to(experiment.root)): p.read_bytes() for p in experiment.root.rglob('*') if p.is_file()}
    runner, backend = make_runner(experiment)
    with pytest.raises(ValueError, match='existing supplementary'):
        runner.run([stage])
    after = {str(p.relative_to(experiment.root)): p.read_bytes() for p in experiment.root.rglob('*') if p.is_file()}
    assert before == after
    assert not backend.calls


def test_distinct_invocations_keep_each_stage_status_unchanged(experiment):
    runner, backend = make_runner(experiment)
    runner.run(['regression'])
    regression_status = experiment.root / 'bonus_results/run_status_regression.json'
    prior_bytes = regression_status.read_bytes()
    runner.run(['b1'])
    assert regression_status.read_bytes() == prior_bytes
    b1 = json.loads((experiment.root / 'bonus_results/run_status_b1.json').read_text())
    assert b1['stage'] == 'b1' and b1['result']['status'] == 'complete'
    assert b1['seal_unchanged'] is True
    assert json.loads(regression_status.read_text())['result']['status'] == 'complete'


def test_b4_second_attempt_preserves_complete_rank_evidence(experiment):
    runner, backend = make_runner(experiment)
    runner.run(['b4'])
    path = experiment.root / 'bonus_results/rank_sweep.json'
    original = path.read_bytes()
    calls = list(backend.calls)
    with pytest.raises(ValueError, match='existing supplementary'):
        runner.run(['b4'])
    assert path.read_bytes() == original and backend.calls == calls
