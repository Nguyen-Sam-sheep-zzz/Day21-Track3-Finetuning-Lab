#!/usr/bin/env python3
"""Supplementary GPU evidence only; never rewrites the sealed Lab 21 experiment.

python student_tools/deadline_experiments.py --stages regression,b1,b4 \
    --stop-at-utc 2026-10-07T16:20:00Z

Run from the restored, original Colab T4 workspace. The clock guards are admission
checks, not a promise to terminate an in-flight CUDA call at an exact wall time.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import gc
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from labkit import config, data, evidence, evaluate as ev, generate, modeling, train

PACKAGES = ('torch', 'transformers', 'tokenizers', 'peft', 'trl', 'accelerate',
            'datasets', 'bitsandbytes', 'unsloth', 'safetensors', 'pytest')


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def seal(root):
    """Content inventory includes original adapters, datasets, results and source."""
    paths = []
    for folder in ('results', 'adapters', 'data', 'src/labkit', 'notebooks'):
        paths.extend(p for p in (root / folder).rglob('*') if p.is_file()
                     and '__pycache__' not in p.parts and p.suffix != '.pyc')
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(paths)}


def package_versions():
    versions = {'python': platform.python_version()}
    for package in PACKAGES:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'not installed'
    return versions


class GPUBackend:
    """Lazy imports keep provenance/deadline checks and CPU tests GPU independent."""
    def __init__(self, root, tier):
        self.root, self.tier = root, tier
        self.held = []

    def _base(self):
        from labkit import device
        if device.precision() != 'fp16':
            raise ValueError('follow-up requires the original fp16 T4 recipe')
        model, tok = generate.load_base(self.tier, load_in_4bit=False)
        evidence.verify_model_revisions(self.root, model, tok)
        self.held.extend([model, tok])
        return model, tok

    def load_adapter(self, path):
        from peft import PeftModel
        model, tok = self._base()
        model = PeftModel.from_pretrained(model, str(path))
        model.eval()
        self.held.append(model)
        return model, tok

    def generate(self, model, tok, prompts, **kwargs):
        return generate.generate_batch(model, tok, prompts, **kwargs)

    def merge(self, model):
        merged = model.merge_and_unload(safe_merge=True)
        merged.eval()
        self.held.append(merged)
        return merged

    def load_hot_swap(self, paths):
        from peft import PeftModel
        base, tok = self._base()
        model = PeftModel.from_pretrained(base, str(paths['correct']), adapter_name='correct')
        model.load_adapter(str(paths['attn_only']), adapter_name='attn_only')
        model.eval()
        self.held.append(model)
        # Runtime identity documents one base object for both activations, not a signature.
        identity = f'{type(base).__name__}:{id(base)}'
        self.hot_base = model.get_base_model()
        return model, tok, identity

    def select_adapter(self, model, key):
        if model.get_base_model() is not self.hot_base:
            raise ValueError('hot-swap base identity changed')
        model.set_adapter(key)
        model.eval()
        if model.active_adapter != key:
            raise ValueError('hot-swap did not activate requested adapter')

    def train_rank(self, root, rank, train_rows):
        from datasets import Dataset
        from peft import LoraConfig
        from trl import SFTConfig, SFTTrainer
        from transformers import set_seed
        set_seed(42)
        out = root / 'bonus_adapters' / f'rank_{rank}'
        if out.exists() and any(out.iterdir()):
            raise ValueError(f'bonus adapter already exists: {out}; preserve it in a separate folder before rerun')
        model, tok = self._base()
        spec = dataclasses.replace(config.SPECS['correct'], key=f'rank_{rank}', r=rank, alpha=2*rank)
        targets = modeling.resolve_target_modules(model, 'text-linear')
        trainable = modeling.count_lora_params(model, targets, rank)
        rows = data.to_training_dataset(tok, train_rows, max_length=1024, mask_mode='assistant-only')
        if len(rows) != 225 or train.planned_steps(len(rows), self.tier, 2) != 30:
            raise ValueError('tokenized data/step budget drift from the original 225-row recipe')
        supervised = sum(sum(label != data.IGNORE_INDEX for label in row['labels']) for row in rows)
        total = sum(len(row['labels']) for row in rows)
        if not 0 < supervised < total:
            raise ValueError('assistant-only supervision mask is empty or covers all tokens')
        want = train.sft_config_kwargs(self.tier, spec, str(out), max_steps=30,
                                       num_train_epochs=2, mask_mode='assistant-only',
                                       seed=42, precision='fp16')
        sft, dropped = train.filter_kwargs(SFTConfig, want, label=f'bonus-r{rank}/SFTConfig')
        required = {'max_steps', 'learning_rate', 'packing', 'fp16', 'bf16', 'seed'}
        if required.intersection(dropped):
            raise ValueError(f'installed TRL drops required recipe fields: {required.intersection(dropped)}')
        lora, lora_dropped = train.filter_kwargs(LoraConfig, train.lora_config_kwargs(spec, targets), label='bonus/LoraConfig')
        if lora_dropped:
            raise ValueError(f'installed PEFT drops adapter recipe fields: {lora_dropped}')
        model.config.use_cache = False
        trainer = SFTTrainer(model=model, args=SFTConfig(**sft),
                             train_dataset=Dataset.from_list(rows), processing_class=tok,
                             peft_config=LoraConfig(**lora))
        self.held.append(trainer)
        fix = train.align_trainable_precision(trainer.model, precision='fp16')
        print(f'rank={rank}; precision alignment={fix}', flush=True)
        generate.free_memory()
        started = time.perf_counter()
        result = trainer.train()
        elapsed = time.perf_counter() - started
        trainer.model.save_pretrained(out)
        tok.save_pretrained(out)
        row = train.summarize_run(spec, self.tier, targets, trainable, elapsed, generate.peak_vram_gb())
        row.update(final_loss=float(result.training_loss), mask_mode='assistant-only', max_steps=30,
                   epochs=2, seed=42, max_length=1024, precision_fix=fix,
                   model_revision=getattr(model.config, '_commit_hash', None),
                   tokenizer_revision=tok.init_kwargs.get('_commit_hash'), dropped_sft_fields=dropped,
                   training_log_history=trainer.state.log_history)
        # Release optimizer state before target-only inference.
        adapter = trainer.model
        self.held.remove(trainer)
        del trainer
        adapter.config.use_cache = True
        if hasattr(adapter, 'gradient_checkpointing_disable'):
            adapter.gradient_checkpointing_disable()
        adapter.eval()
        self.held.append(adapter)
        generate.free_memory()
        return adapter, tok, row

    def release(self):
        self.held.clear()
        if hasattr(self, 'hot_base'):
            del self.hot_base
        gc.collect()
        generate.free_memory()


class Runner:
    def __init__(self, root, *, backend=None, now=None, stop_at=None):
        self.root = Path(root)
        self.tier = config.get_tier('T4')
        self.backend = backend or GPUBackend(self.root, self.tier)
        self.now = now or (lambda: datetime.now(timezone.utc))
        self.stop_at = stop_at or datetime(2026, 10, 7, 16, 20, tzinfo=timezone.utc)
        if self.stop_at.tzinfo is None:
            raise ValueError('stop_at requires an explicit UTC timezone')
        self.output = self.root / 'bonus_results'

    def preflight(self):
        self.target = read_jsonl(self.root / 'data/eval_target.jsonl')
        self.regression = read_jsonl(self.root / 'data/eval_regression.jsonl')
        self.train_rows = read_jsonl(self.root / 'data/split/train.jsonl')
        if (len(self.target), len(self.regression), len(self.train_rows),
            len(read_jsonl(self.root / 'data/split/val.jsonl')),
            len(read_jsonl(self.root / 'data/train_seed.jsonl'))) != (50, 15, 225, 25, 250):
            raise ValueError('requires original full 50/15 evaluation and 250 -> 225/25 corpus')
        ctx = evidence.experiment_context(self.root, self.tier, precision='fp16', seed=42)
        self.raw = evidence.verify_baseline(self.root, ctx, self.target, self.regression)
        self.manifest = json.loads((self.root / 'results/experiment_manifest.json').read_text(encoding='utf-8'))
        self.correct = self.root / 'adapters/correct'
        self.attn = self.root / 'adapters/attn_only'
        for path in (self.correct, self.attn):
            evidence.verify_adapter_provenance(self.root, path, load_in_4bit=False)
            if not (path / 'adapter_model.safetensors').is_file():
                raise ValueError(f'actual adapter weights missing: {path}')
        with (self.root / 'results/runs.csv').open(encoding='utf-8', newline='') as fh:
            runs = list(csv.DictReader(fh))
        self.original_correct = [r for r in runs if r['run'] == 'correct'][-1]
        expected = {'r': '16', 'lora_alpha': '32', 'placement': 'text-linear', 'mask_mode': 'assistant-only', 'max_steps': '30'}
        if any(str(self.original_correct.get(key)) != value for key, value in expected.items()) or float(self.original_correct['learning_rate']) != .0001:
            raise ValueError('original correct training recipe does not match fixed-rank sweep')
        adapter = json.loads((self.correct / 'adapter_config.json').read_text(encoding='utf-8'))
        if adapter.get('r') != 16 or adapter.get('lora_alpha') != 32:
            raise ValueError('original correct adapter rank/alpha mismatch')
        self.original_verdict = json.loads((self.root / 'results/verdict.json').read_text(encoding='utf-8'))
        current = package_versions()
        expected_versions = self.manifest['environment']
        git = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=self.root, capture_output=True, text=True, check=False)
        self.metadata = {
            'script': 'student_tools/deadline_experiments.py',
            'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_commit': git.stdout.strip() if git.returncode == 0 else None,
            'original_source_commit': self.manifest['source_commit'],
            'created_at_utc': self.now().isoformat(), 'stop_at_utc': self.stop_at.isoformat(),
            'context': ctx, 'environment': current, 'expected_environment': expected_versions,
            'package_version_comparison': {key: {'expected': value, 'actual': current.get(key), 'match': value == current.get(key)} for key, value in expected_versions.items()},
            'original_model_revision': self.manifest.get('model_revision'),
            'original_tokenizer_revision': self.manifest.get('tokenizer_revision'),
            'revision_caution': ('Original model/tokenizer revision unavailable; ID and artifact checks cannot prove an identical moving remote revision.' if self.manifest.get('model_revision') is None or self.manifest.get('tokenizer_revision') is None else 'Available original revisions are checked before generation.'),
            'generation': {'do_sample': False, 'enable_thinking': False, 'batch_size': 4},
            'claim_scope': 'supplementary follow-up; original aggregate verdict and baseline files remain sealed',
        }

    def allowed(self, seconds):
        return (self.stop_at - self.now()).total_seconds() >= seconds

    def write(self, name, payload):
        self.output.mkdir(parents=True, exist_ok=True)
        payload = {**payload, 'metadata': self.metadata}
        path = self.output / name
        temporary = path.with_suffix(path.suffix + '.tmp')
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        temporary.replace(path)

    def loaded_metadata(self, model, tok):
        evidence.verify_model_revisions(self.root, model, tok)
        return {'model_revision': getattr(getattr(model, 'config', None), '_commit_hash', None),
                'tokenizer_revision': getattr(tok, 'init_kwargs', {}).get('_commit_hash'),
                'tokenizer': getattr(tok, 'name_or_path', None)}

    def predictions(self, model, tok, rows, *, regression=False, label='follow-up'):
        prompts = [r['instruction'] if regression else r['input'] for r in rows]
        preds, latency = self.backend.generate(model, tok, prompts,
            system=None if regression else config.NAIVE_PROMPT,
            max_new_tokens=96 if regression else 160, enable_thinking=False,
            batch_size=4, label=label)
        if len(preds) != len(rows):
            raise ValueError(f'prediction length mismatch: expected {len(rows)}, got {len(preds)}')
        return preds, latency

    def target_result(self, model, tok, label):
        preds, latency = self.predictions(model, tok, self.target, label=label)
        rows = [{'i': i, 'ticket': row['input'], 'label': row['label'], 'prediction': pred,
                 'target_score': ev.triage_field_accuracy(pred, row['label']),
                 'format_score': float(ev.has_required_keys(pred, ev.TRIAGE_KEYS))}
                for i, (row, pred) in enumerate(zip(self.target, preds))]
        return {'n': 50, 'target': sum(r['target_score'] for r in rows)/50,
                'format': sum(r['format_score'] for r in rows)/50,
                'latency_ms': latency, 'rows': rows,
                'generation': {'system': config.NAIVE_PROMPT, 'max_new_tokens': 160, 'enable_thinking': False, 'do_sample': False, 'batch_size': 4},
                'loaded': self.loaded_metadata(model, tok)}

    def run_regression(self):
        model, tok = self.backend.load_adapter(self.correct)
        loaded = self.loaded_metadata(model, tok)
        preds, latency = self.predictions(model, tok, self.regression, regression=True, label='correct/regression-followup')
        rows = []
        for i, (row, baseline, pred) in enumerate(zip(self.regression, self.raw['regression'], preds)):
            base_score = ev.keyword_recall(baseline['baseline_b_pred'], row['keywords'])
            score = ev.keyword_recall(pred, row['keywords'])
            rows.append({'i': i, 'instruction': row['instruction'], 'keywords': row['keywords'],
                         'baseline_b_pred': baseline['baseline_b_pred'], 'ft_pred': pred,
                         'baseline_b_score': base_score, 'ft_score': score, 'delta': score-base_score})
        original = next(r['regression'] for r in self.original_verdict['comparison'] if r['run'].startswith('(c)'))
        original_exact = (sum(r['baseline_b_score'] for r in rows)/15 + self.original_verdict['verdict']['regression_delta']
                          if 'regression_delta' in self.original_verdict['verdict'] else None)
        self.write('regression_followup.json', {'status': 'complete', 'supplementary_repeat': True,
            'n': 15, 'rows': rows, 'regression': sum(r['ft_score'] for r in rows)/15,
            'baseline_b_regression': sum(r['baseline_b_score'] for r in rows)/15,
            'original_ft_regression_aggregate': original,
            'original_ft_regression_recovered_from_gate_delta': original_exact,
            'generation': {'system': None, 'max_new_tokens': 96, 'enable_thinking': False, 'do_sample': False, 'batch_size': 4},
            'original_aggregate_precision': 'value exactly as stored in original verdict; raw original FT regression predictions were not retained',
            'latency_ms': latency, 'loaded': loaded})
        return {'status': 'complete', 'artifacts': ['regression_followup.json']}

    def run_b1(self):
        # A complete merge check requires two full sweeps; never score a small slice.
        if not self.allowed(300):
            return {'status': 'incomplete', 'reason': 'insufficient budget for full before/after merge'}
        model, tok = self.backend.load_adapter(self.correct)
        self.loaded_metadata(model, tok)
        before = self.target_result(model, tok, 'correct/before-merge')
        model = self.backend.merge(model)
        after = self.target_result(model, tok, 'correct/after-merge')
        delta = after['target'] - before['target']
        self.write('merge_check.json', {'status': 'complete', 'method': 'PEFT merge_and_unload(safe_merge=True), original fp16 base',
            'before': before, 'after': after, 'target_delta': delta, 'tolerance': .01,
            'passed': abs(delta) <= .01, 'merged_weights_saved': False})
        del model, tok
        self.backend.release()
        if not self.allowed(300):
            return {'status': 'partial', 'reason': 'merge completed; insufficient budget for two-adapter hot-swap', 'artifacts': ['merge_check.json']}
        model, tok, identity = self.backend.load_hot_swap({'correct': self.correct, 'attn_only': self.attn})
        self.loaded_metadata(model, tok)
        results = {}
        for key in ('correct', 'attn_only'):
            self.backend.select_adapter(model, key)
            results[key] = {**self.target_result(model, tok, f'hot-swap/{key}'),
                            'active_adapter': key, 'base_identity': identity,
                            'adapter_sha256': hashlib.sha256((self.root / 'adapters' / key / 'adapter_model.safetensors').read_bytes()).hexdigest()}
        self.write('hot_swap.json', {'status': 'complete', 'same_base_identity': identity,
                                     'base_load_count': 1, 'adapters': results,
                                     'method': 'one fresh base; load_adapter then set_adapter; both actual full target sweeps'})
        return {'status': 'complete', 'artifacts': ['merge_check.json', 'hot_swap.json']}

    def append_bonus_run(self, rank, metrics):
        fields = ['run', 'r', 'lora_alpha', 'placement', 'learning_rate', 'max_steps',
                  'final_loss', 'train_seconds', 'peak_vram_gb', 'target', 'format',
                  'reused_original_adapter', 'source_commit']
        path = self.output / 'bonus_runs.csv'
        exists = path.exists()
        with path.open('a', encoding='utf-8', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            if not exists:
                writer.writeheader()
            writer.writerow({'run': f'rank_{rank}', 'r': rank, 'lora_alpha': 2*rank,
                'placement': 'text-linear', 'learning_rate': .0001, 'max_steps': 30,
                **{k: metrics.get(k) for k in fields if k in metrics}, 'source_commit': self.metadata['source_commit']})

    def run_b4(self):
        sweep = {'status': 'partial', 'ranks': {}, 'fixed_recipe': {'ranks': [8, 16, 64],
                 'lora_alpha': '2r', 'placement': 'text-linear', 'learning_rate': .0001,
                 'max_steps': 30, 'epochs': 2, 'seed': 42, 'max_length': 1024,
                 'mask_mode': 'assistant-only', 'n_target': 50, 'precision': 'fp16'},
                 'ranking_basis': 'target and format; training loss retained as descriptive evidence'}
        # Reusing the r16 adapter requires a fresh 50-item score, no new training.
        for rank in (16, 8, 64):
            seconds = 150 if rank == 16 else 900
            if not self.allowed(seconds):
                sweep['reason'] = f'insufficient budget to start rank {rank} ({seconds}s admission reserve)'
                self.write('rank_sweep.json', sweep)
                return {'status': 'partial' if sweep['ranks'] else 'incomplete', 'reason': sweep['reason'], 'artifacts': ['rank_sweep.json']}
            print(f'B4 rank {rank}: full recipe and full 50-target score', flush=True)
            if rank == 16:
                model, tok = self.backend.load_adapter(self.correct)
                metrics = dict(self.original_correct)
                metrics.update(reused_original_adapter=True, training_metrics_source='original results/runs.csv; historical training time/VRAM')
            else:
                model, tok, metrics = self.backend.train_rank(self.root, rank, self.train_rows)
                metrics['reused_original_adapter'] = False
            self.loaded_metadata(model, tok)
            score = self.target_result(model, tok, f'rank_{rank}/target')
            result = {**metrics, **score, 'r': rank, 'lora_alpha': 2*rank,
                      'learning_rate': .0001, 'placement': 'text-linear', 'max_steps': 30}
            sweep['ranks'][str(rank)] = result
            if len(sweep['ranks']) == 3:
                sweep['status'] = 'complete'
            self.write('rank_sweep.json', sweep)
            self.append_bonus_run(rank, result)
            del model, tok
            self.backend.release()
        return {'status': 'complete', 'artifacts': ['rank_sweep.json', 'bonus_runs.csv']}

    def run(self, stages):
        if not stages or len(stages) != len(set(stages)) or any(stage not in ('regression', 'b1', 'b4') for stage in stages):
            raise ValueError('stages must be distinct members of regression,b1,b4')
        self.preflight()  # No GPU loading before frozen context and adapter checks.
        self.original_seal = seal(self.root)
        self.metadata['sealed_inventory_sha256'] = hashlib.sha256(json.dumps(self.original_seal, sort_keys=True).encode()).hexdigest()
        status = {'stages': {}, 'seal_unchanged': None}
        try:
            for stage in stages:
                if not self.allowed(120 if stage == 'regression' else 150):
                    status['stages'][stage] = {'status': 'incomplete', 'reason': 'deadline or insufficient minimum admission reserve'}
                    continue
                print(f'FOLLOW-UP {stage}; cutoff={self.stop_at.isoformat()}', flush=True)
                try:
                    status['stages'][stage] = getattr(self, f'run_{stage}')()
                except Exception as exc:
                    status['stages'][stage] = {'status': 'failed', 'error': f'{type(exc).__name__}: {exc}'}
                    raise
                finally:
                    self.backend.release()
                self.write('run_status.json', status)
        finally:
            status['seal_unchanged'] = seal(self.root) == self.original_seal
            self.write('run_status.json', status)
            if not status['seal_unchanged']:
                raise ValueError('sealed original artifacts changed during supplementary experiment')
        return status


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stages', default='regression,b1,b4')
    parser.add_argument('--stop-at-utc', default='2026-10-07T16:20:00Z')
    parser.add_argument('--root', type=Path, default=ROOT)
    args = parser.parse_args(argv)
    stop_at = datetime.fromisoformat(args.stop_at_utc.replace('Z', '+00:00'))
    status = Runner(args.root, stop_at=stop_at).run([s.strip() for s in args.stages.split(',')])
    print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
    return 0 if all(s['status'] == 'complete' for s in status['stages'].values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
