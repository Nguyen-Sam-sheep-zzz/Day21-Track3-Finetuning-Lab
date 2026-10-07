"""Pre-training provenance and exact paired evaluation; never performs inference.

The manifest hashes its companion artifacts. Explicit pre-training reruns archive the
complete previous attempt; trained adapters or training rows prohibit replacement.
This detects accidental drift, not adversarial rewriting or deleted training history.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import math
import pathlib
import platform
import subprocess
import uuid
from datetime import datetime, timezone

from . import config, evaluate as ev

ARTIFACTS = ('baselines_frozen.json', 'baseline_predictions.json', 'environment.txt')
MANIFEST = 'experiment_manifest.json'


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _json(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False)


def experiment_context(root: pathlib.Path, tier: config.Tier, *, precision: str,
                       eval_limit: int = 0, seed: int = 42) -> dict:
    """Identity checked again in NB5; package versions/commit are descriptive metadata."""
    if eval_limit < 0:
        raise ValueError('eval_limit must be nonnegative')
    root = pathlib.Path(root)
    names = ['data/eval_target.jsonl', 'data/eval_regression.jsonl', 'data/train_seed.jsonl']
    names += [f'data/split/{name}.jsonl' for name in ('train', 'val')
              if (root / f'data/split/{name}.jsonl').exists()]
    hashes = {name: _sha((root / name).read_bytes()) for name in names}
    counts = []
    for name in names[:2]:
        n = sum(bool(line.strip()) for line in (root / name).read_text(encoding='utf-8').splitlines())
        counts.append(min(n, eval_limit) if eval_limit else n)
    return {
        'model': tier.model_id, 'tier': tier.name, 'seed': seed,
        'max_length': tier.max_length, 'epochs': config.training_epochs(),
        'precision': precision, 'eval_limit': eval_limit or None,
        'n_target': counts[0], 'n_regression': counts[1], 'smoke_mode': bool(eval_limit),
        'optimized_prompt_sha': _sha(config.OPTIMIZED_PROMPT.encode('utf-8')),
        'naive_prompt_sha': _sha(config.NAIVE_PROMPT.encode('utf-8')),
        'file_sha256': hashes,
    }


def _read_bundle(results: pathlib.Path) -> tuple[dict, dict, dict]:
    try:
        manifest = json.loads((results / MANIFEST).read_text(encoding='utf-8'))
        if manifest.get('schema_version') != 1 or set(manifest['artifact_sha256']) != set(ARTIFACTS):
            raise ValueError('invalid evidence manifest schema')
        for name in ARTIFACTS:
            if _sha((results / name).read_bytes()) != manifest['artifact_sha256'][name]:
                raise ValueError(f'evidence hash mismatch: {name}')
        raw = json.loads((results / 'baseline_predictions.json').read_text(encoding='utf-8'))
        frozen = json.loads((results / 'baselines_frozen.json').read_text(encoding='utf-8'))
        _validate_frozen(manifest['context'], raw, frozen)
        return manifest, raw, frozen
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'incomplete or invalid baseline evidence: {exc}') from exc


def assert_baseline_writable(root: pathlib.Path, *, replace_before_training: bool = False) -> None:
    """Run BEFORE NB2 loads a model and again immediately before recording."""
    root = pathlib.Path(root)
    if any(p.is_file() and p.name != '.gitkeep' for p in (root / 'adapters').rglob('*')):
        raise ValueError('training artifacts exist; baseline evidence cannot be replaced')
    run_log = root / 'results' / 'runs.csv'
    if run_log.exists():
        with run_log.open(encoding='utf-8', newline='') as fh:
            if list(csv.DictReader(fh)):
                raise ValueError('training rows exist; baseline evidence cannot be replaced')
    results = root / 'results'
    if any((results / name).exists() for name in (*ARTIFACTS, MANIFEST)):
        _read_bundle(results)  # Explicit rerun cannot quietly erase inconsistent evidence.
        if not replace_before_training:
            raise ValueError('baseline exists; explicit pre-training rerun is required (BASELINE_RERUN=1)')


def _check_target_identity(target: list[dict], saved: list[dict]) -> None:
    if not target or len(target) != len(saved):
        raise ValueError('target prediction length mismatch or empty eval')
    for i, (row, old) in enumerate(zip(target, saved)):
        if old['i'] != i or old['ticket'] != row['input'] or old['label'] != row['label']:
            raise ValueError(f'target order/input/label mismatch at {i}')


def _check_regression_identity(regression: list[dict], saved: list[dict]) -> None:
    if not regression or len(regression) != len(saved):
        raise ValueError('regression prediction length mismatch or empty eval')
    for i, (row, old) in enumerate(zip(regression, saved)):
        if old['i'] != i or old['instruction'] != row['instruction'] or old['keywords'] != row['keywords']:
            raise ValueError(f'regression order/input/label mismatch at {i}')


def _validate_frozen(context: dict, raw: dict, frozen: dict) -> None:
    for name in ('model', 'tier', 'n_target', 'n_regression', 'eval_limit', 'smoke_mode'):
        if frozen.get(name) != context[name]:
            raise ValueError(f'frozen context mismatch: {name}')
    if frozen.get('optimized_prompt_sha') != context['optimized_prompt_sha'][:16]:
        raise ValueError('frozen optimized prompt mismatch')
    if len(raw['target']) != context['n_target'] or len(raw['regression']) != context['n_regression']:
        raise ValueError('prediction length mismatch with context')
    # Re-score raw output with the unchanged lab scorers. Stored scores are never trusted.
    _check_target_identity([{'input': r['ticket'], 'label': r['label']} for r in raw['target']], raw['target'])
    _check_regression_identity([{'instruction': r['instruction'], 'keywords': r['keywords']}
                               for r in raw['regression']], raw['regression'])
    for key in ('a', 'b'):
        scores = frozen[f'baseline_{key}']
        predictions = [r[f'baseline_{key}_pred'] for r in raw['target']]
        expected = {
            'target': sum(ev.triage_field_accuracy(p, r['label']) for p, r in zip(predictions, raw['target'])) / len(predictions),
            'format': sum(ev.has_required_keys(p, ev.TRIAGE_KEYS) for p in predictions) / len(predictions),
            'regression': sum(ev.keyword_recall(r[f'baseline_{key}_pred'], r['keywords'])
                              for r in raw['regression']) / len(raw['regression']),
        }
        if scores['n'] != len(predictions):
            raise ValueError('frozen score count mismatch')
        for metric, value in expected.items():
            if not math.isclose(scores[metric], value, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f'frozen score mismatch: baseline_{key}.{metric}')


def _revisions(model, tokenizer) -> dict:
    return {
        'model_revision': getattr(getattr(model, 'config', None), '_commit_hash', None),
        'tokenizer_revision': getattr(tokenizer, 'init_kwargs', {}).get('_commit_hash'),
        'tokenizer': getattr(tokenizer, 'name_or_path', None),
    }


def record_baseline(root: pathlib.Path, context: dict, target: list[dict], regression: list[dict],
                    preds_a: list[str], preds_b: list[str], rpreds_a: list[str], rpreds_b: list[str],
                    frozen: dict, *, model=None, tokenizer=None,
                    replace_before_training: bool = False) -> dict:
    """Save outputs already produced by NB2; explicit reruns preserve every prior byte."""
    root = pathlib.Path(root)
    assert_baseline_writable(root, replace_before_training=replace_before_training)
    if len(target) != len(preds_a) or len(target) != len(preds_b):
        raise ValueError('target prediction length mismatch')
    if len(regression) != len(rpreds_a) or len(regression) != len(rpreds_b):
        raise ValueError('regression prediction length mismatch')
    raw = {
        'target': [{'i': i, 'ticket': r['input'], 'label': r['label'],
                    'baseline_a_pred': pa, 'baseline_b_pred': pb,
                    'baseline_a_score': ev.triage_field_accuracy(pa, r['label']),
                    'baseline_b_score': ev.triage_field_accuracy(pb, r['label'])}
                   for i, (r, pa, pb) in enumerate(zip(target, preds_a, preds_b))],
        'regression': [{'i': i, 'instruction': r['instruction'], 'keywords': r['keywords'],
                        'baseline_a_pred': pa, 'baseline_b_pred': pb,
                        'baseline_a_score': ev.keyword_recall(pa, r['keywords']),
                        'baseline_b_score': ev.keyword_recall(pb, r['keywords'])}
                       for i, (r, pa, pb) in enumerate(zip(regression, rpreds_a, rpreds_b))],
    }
    _validate_frozen(context, raw, frozen)
    versions = {'python': platform.python_version()}
    for package in ('torch', 'transformers', 'tokenizers', 'peft', 'trl', 'accelerate',
                    'datasets', 'bitsandbytes', 'unsloth', 'safetensors', 'pytest'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    commit = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=root, capture_output=True,
                            text=True, check=False)
    payloads = {'baselines_frozen.json': _json(frozen), 'baseline_predictions.json': _json(raw),
                'environment.txt': '\n'.join(f'{k}=={v}' for k, v in versions.items()) + '\n'}
    manifest = {
        'schema_version': 1, 'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_commit': commit.stdout.strip() if commit.returncode == 0 else None,
        'context': context, 'environment': versions, **_revisions(model, tokenizer),
        'artifact_sha256': {name: _sha(body.encode('utf-8')) for name, body in payloads.items()},
    }
    payloads[MANIFEST] = _json(manifest)
    results = root / 'results'
    results.mkdir(parents=True, exist_ok=True)
    if (results / MANIFEST).exists():
        archive = results / 'baseline_history' / uuid.uuid4().hex
        archive.mkdir(parents=True)
        for name in (*ARTIFACTS, MANIFEST):
            (results / name).rename(archive / name)
    # Exclusive creation prevents a concurrent run from silently overwriting evidence.
    # Manifest is the completion marker; interrupted writes fail validation on the next run.
    for name, body in payloads.items():
        with (results / name).open('x', encoding='utf-8', newline='\n') as fh:
            fh.write(body)
    return manifest


def verify_baseline(root: pathlib.Path, context: dict, target: list[dict], regression: list[dict]) -> dict:
    """Fail before FT inference if files, config, prompts or ordered eval identity changed."""
    manifest, raw, _ = _read_bundle(pathlib.Path(root) / 'results')
    if manifest['context'] != context:
        raise ValueError('baseline context mismatch: model/tier/config/prompt/eval hashes changed')
    _check_target_identity(target, raw['target'])
    _check_regression_identity(regression, raw['regression'])
    return raw


def verify_model_revisions(root: pathlib.Path, model, tokenizer) -> None:
    """After model loading but before generation, require any recorded resolved revision."""
    manifest, _, _ = _read_bundle(pathlib.Path(root) / 'results')
    current = _revisions(model, tokenizer)
    for name in ('model_revision', 'tokenizer_revision', 'tokenizer'):
        if manifest[name] is not None and manifest[name] != current[name]:
            raise ValueError(f'baseline model/tokenizer revision mismatch: {name}')


def pair_qualitative(target: list[dict], baseline_rows: list[dict], ft_preds: list[str]) -> list[dict]:
    """Full untruncated outputs paired by validated index, exact ticket and exact label."""
    _check_target_identity(target, baseline_rows)
    if len(ft_preds) != len(target):
        raise ValueError('fine-tune prediction length mismatch')
    paired = []
    for i, (row, bp, fp) in enumerate(zip(target, baseline_rows, ft_preds)):
        bscore = ev.triage_field_accuracy(bp['baseline_b_pred'], row['label'])
        fscore = ev.triage_field_accuracy(fp, row['label'])
        paired.append({'i': i, 'ticket': row['input'], 'label': row['label'],
                       'baseline_b_pred': bp['baseline_b_pred'], 'ft_pred': fp,
                       'baseline_b_score': bscore, 'ft_score': fscore, 'delta': fscore - bscore})
    return paired


def summarize_pairs(paired: list[dict]) -> dict:
    """Choose real losses, wins and a tie; report scarce/zero losses honestly."""
    losses = [r for r in paired if r['delta'] < 0]
    wins = [r for r in paired if r['delta'] > 0]
    ties = [r for r in paired if r['delta'] == 0]
    selected = losses[:2] + wins[:2] + ties[:1]
    selected_ids = {r['i'] for r in selected}
    selected += [r for r in paired if r['i'] not in selected_ids][:max(0, 5 - len(selected))]
    return {'losses': len(losses), 'wins': len(wins), 'ties': len(ties), 'total': len(paired),
            'two_loss_examples_available': len(losses) >= 2,
            'five_examples_available': len(paired) >= 5, 'selected_examples': selected}
