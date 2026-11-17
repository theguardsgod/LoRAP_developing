"""Self-contained paired GraphLoRA repair experiments for another GPU server.

Train fresh None/GraphLoRA pairs; no historical checkpoints or local paths are
required. All validation choices finish before the one-shot test barrier opens.
"""
import argparse
import csv
import fcntl
import io
import json
import math
import os
from pathlib import Path
import statistics
import time
import traceback

import torch
from torch.nn import functional as F

from ..graphlora import OBJECTIVE_VERSION, checked_gradient_norm
from ..table1_frozen.backends import build, load_state, optimizer, predict, project
from ..table1_frozen.common import (config_hash, consider, digest, finite_gradients,
    restore_rng, rng_state, save, seed_all, write_json)
from ..table1_frozen.data import Bundle, labels_and_output
from ..table1_frozen.train import task_loss
from .model import GraphLoRA, counts, make_optimizer, objective
from .run import HP, read, require_current_objective, sha, snapshot, verify_sources

PROFILES = {
    'cora': ['QAT_Cora_GIN', 'QAT_Cora_GCN', 'QAT_Cora_GAT'],
    'focal': ['QAT_Cora_GIN', 'QAT_Cora_GCN', 'QAT_Cora_GAT', 'QAT_MNIST_GCN', 'QAT_MNIST_GAT'],
}
VARIANTS = ('none', 'graphlora')


def validate_study(study):
    if study.get('adaptation', 'graphlora') == 'graphlora':
        require_current_objective(study)
    elif study.get('adaptation') == 'gpf_lorap':
        hp = study['hyperparameters']
        if study.get('objective_version') != 1 or study['protocol'] != 'joint':
            raise ValueError('Unsupported GPF-LoRAP experiment protocol')
        if any(isinstance(hp[k], bool) or not isinstance(hp[k], int) or hp[k] < 1 for k in ('rank', 'num_prompts')):
            raise ValueError('Positive integer GPF-LoRAP rank and prompt count required')
        if not math.isfinite(hp['gradient_norm_limit']) or hp['gradient_norm_limit'] <= 0:
            raise ValueError('Finite positive gradient limit required')
    else:
        raise ValueError('Unknown adaptation')


def adapter_api(variant):
    if variant == 'graphlora':
        from . import model
    elif variant == 'gpf_lorap':
        from ..table1_gpf_rerun import model
    else:
        raise ValueError('Unknown adaptation variant')
    return model


def prepare(root, data_root, cells=None, seeds=None, smoke=False, normalization_floor=.1,
            adaptation='graphlora', cell_configs=None):
    from ..table1_frozen.registry import CELLS
    registry = CELLS if cell_configs is None else cell_configs
    root, data_root = Path(root).resolve(), Path(data_root).resolve()
    if root.exists():
        raise FileExistsError('Use a new output directory or resume with worker')
    cells = PROFILES['cora'] if cells is None else cells
    seeds = ([41] if smoke else [41, 42, 43]) if seeds is None else seeds
    if not cells or len(set(cells)) != len(cells) or set(cells) - set(registry):
        raise ValueError('Choose unique registered cells')
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Choose unique seeds')
    if adaptation == 'gpf_lorap':
        hp = dict(rank=2, num_prompts=20, gradient_norm_limit=1e6,
                  prompt_lr='backbone_lr', prompt_weight_decay=0.,
                  sharing='score_per_width_factors_per_layer', insertion='before_native_update')
    else:
        hp = dict(HP, normalization_floor=normalization_floor)
    variants = ('none', adaptation)
    study = dict(portable_version=1, objective_version=1 if adaptation == 'gpf_lorap' else OBJECTIVE_VERSION,
                 adaptation=adaptation, method='GPF-LoRAP' if adaptation == 'gpf_lorap' else 'GraphLoRA-adapted',
                 protocol='joint', hyperparameters=hp,
                 seeds=seeds, cells=cells, variants=list(variants), smoke=smoke,
                 subset_limit=8 if smoke else None, controls_reused=False,
                 selection='minimum validation task loss, earliest epoch on ties',
                 test_policy='all paired fits complete before one test per selected model',
                 created_unix=time.time(), torch_version=torch.__version__)
    validate_study(study)
    jobs = []
    for cell in cells:
        c = dict(registry[cell])
        c['data_root'] = str(data_root / c['dataset'].lower())
        for seed in seeds:
            for variant in variants:
                jobs.append(dict(cell=cell, config=c, seed=seed, variant=variant,
                                 epochs=2 if smoke else c['baseline_epochs'],
                                 directory=f'{cell}/seed_{seed}/{variant}'))
    study.update(jobs=jobs, training_jobs=len(jobs))
    root.mkdir(parents=True)
    write_json(root / 'study.json', study)
    for job in jobs:
        (root / job['directory']).mkdir(parents=True)
    snapshot(root)
    generate(root, study)
    return study


def job_key(study, job):
    return config_hash(dict(job=job, hyperparameters=study['hyperparameters'],
                            objective_version=study['objective_version'], subset_limit=study['subset_limit']))


def model_for(study, job, bundle, device):
    seed_all(job['seed'])
    base = build(job['config'], bundle, job['seed']).to(device)
    if 'attention_dropout' in job['config']:
        for layer in base.modules():
            if hasattr(layer, 'propagate') and hasattr(layer, 'dropout'):
                layer.dropout = job['config']['attention_dropout']
    initial = digest(base)
    # Consume the same loader setup RNG for both members of a pair.
    example = next(bundle.batches('train', job['seed'], 0, device, limit=2))
    if job['variant'] == 'none':
        return base, initial
    if job['variant'] == 'gpf_lorap':
        model = adapter_api(job['variant']).GPFLoRAP(base, job['config'], job['seed'], example,
                    study['hyperparameters']['rank'], study['hyperparameters']['num_prompts'])
    else:
        model = GraphLoRA(base, job['config'], job['seed'], example, study['hyperparameters']['rank'])
    if digest(base) != initial:
        raise AssertionError('Adapter construction changed the initial backbone')
    return model, initial


def close(model):
    if callable(getattr(model, 'close', None)):
        model.close()


@torch.no_grad()
def evaluate(model, job, study, bundle, device):
    model.eval()
    total_loss = total_metric = count = 0
    for batch in bundle.batches('val', job['seed'], 0, device, limit=study['subset_limit']):
        out, _ = predict(model, job['config'], batch)
        out, y = labels_and_output(job['config'], batch, out, 'val')
        loss = task_loss(job['config'], out, y)
        metric = ((out.flatten() - y).abs().sum() if job['config']['regression'] else
                  (out.argmax(1) == y).sum())
        total_loss += loss.item() * len(y)
        total_metric += metric.item()
        count += len(y)
    return dict(loss=total_loss / count, metric=total_metric / count *
                (1 if job['config']['regression'] else 100), examples=count)


def train_epoch(model, job, study, bundle, device, epoch, opt):
    model.train()
    totals = {key: 0. for key in ('loss', 'contrast', 'structure', 'storage', 'objective', 'metric')}
    count, peak = 0, 0.
    for batch in bundle.batches('train', job['seed'], epoch, device, limit=study['subset_limit']):
        opt.zero_grad(set_to_none=True)
        if job['variant'] != 'none':
            loss, logits, y, parts = adapter_api(job['variant']).objective(model, job['config'], batch, study['hyperparameters'])
        else:
            logits, budget = predict(model, job['config'], batch)
            logits, y = labels_and_output(job['config'], batch, logits, 'train')
            task = task_loss(job['config'], logits, y)
            storage = job['config']['a_loss'] * F.relu(budget - job['config']['a_storage']).square()
            loss = task + storage
            parts = dict(task=task, contrast=task.new_zeros(()), structure=task.new_zeros(()), storage=storage)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training objective')
        loss.backward()
        finite_gradients(model)
        peak = max(peak, checked_gradient_norm(model, study['hyperparameters']['gradient_norm_limit']).item())
        opt.step()
        project(model, job['config'])
        metric = ((logits.flatten() - y).abs().sum() if job['config']['regression'] else
                  (logits.argmax(1) == y).sum())
        for key, value in [('loss', parts['task']), ('contrast', parts['contrast']),
                           ('structure', parts['structure']), ('storage', parts['storage']), ('objective', loss)]:
            totals[key] += value.item() * len(y)
        totals['metric'] += metric.item()
        count += len(y)
    result = {k: v / count for k, v in totals.items()}
    if not job['config']['regression']:
        result['metric'] *= 100
    return dict(result, examples=count, gradient_norm_max=peak)


def fit(root, study, job, device):
    validate_study(study)
    path, key = root / job['directory'], job_key(study, job)
    if (path / 'done.json').exists():
        if read(path / 'done.json')['config_hash'] != key:
            raise ValueError('Completed configuration changed')
        return
    config = dict(job=job, hyperparameters=study['hyperparameters'], config_hash=key)
    if (path / 'config.json').exists() and read(path / 'config.json') != config:
        raise ValueError('Interrupted configuration changed')
    write_json(path / 'config.json', config)
    bundle = Bundle(job['config'])
    write_json(path / 'split.json', bundle.manifest())
    model, initial = model_for(study, job, bundle, device)
    try:
        opt = (adapter_api(job['variant']).make_optimizer(model, job['config'])
               if job['variant'] != 'none' else optimizer(model, job['config']))
        c = job['config']
        scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=5)
                     if c['framework'] == 'A2Q' and not c['node'] else
                     torch.optim.lr_scheduler.StepLR(opt, 50, .5) if c['framework'] != 'A2Q' else None)
        history, best, start_epoch, peak_memory = [], None, 1, 0
        if (path / 'last.pt').exists():
            last = torch.load(path / 'last.pt', map_location='cpu', weights_only=False)
            if last['config_hash'] != key:
                raise ValueError('Resume configuration changed')
            load_state(model, last['model']); opt.load_state_dict(last['optimizer'])
            if scheduler:
                scheduler.load_state_dict(last['scheduler'])
            history, best, start_epoch = last['history'], last['best'], last['epoch'] + 1
            peak_memory = last['peak_cuda_memory_bytes']
            restore_rng(last['rng'], device)
        if torch.device(device).type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        for epoch in range(start_epoch, job['epochs'] + 1):
            if (root / 'STOP').exists():
                return
            started = time.monotonic()
            training = train_epoch(model, job, study, bundle, device, epoch, opt)
            validation = evaluate(model, job, study, bundle, device)
            previous = best
            best = consider(best, validation['loss'], epoch, model)
            if previous is not best:
                save(path / 'best.pt', best)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(validation['loss'])
            elif scheduler:
                scheduler.step()
            if torch.device(device).type == 'cuda':
                peak_memory = max(peak_memory, torch.cuda.max_memory_allocated(device))
            history.append(dict(epoch=epoch, train=training, validation=validation,
                                seconds=time.monotonic() - started))
            save(path / 'last.pt', dict(config_hash=key, epoch=epoch, model=model.state_dict(),
                 optimizer=opt.state_dict(), scheduler=scheduler.state_dict() if scheduler else None,
                 rng=rng_state(device), best=best, history=history, peak_cuda_memory_bytes=peak_memory))
            write_json(path / 'history.json', history)
            write_json(path / 'progress.json', dict(epoch=epoch, epochs=job['epochs'], best_epoch=best['epoch'], validation=validation))
            if epoch == 1 or epoch % 25 == 0:
                print(job['cell'], job['seed'], job['variant'], epoch, validation, flush=True)
        load_state(model, best['model'])
        validation = evaluate(model, job, study, bundle, device)
        if abs(validation['loss'] - best['validation_loss']) > 1e-5:
            raise AssertionError('Selected validation loss did not replay')
        extra = adapter_api(job['variant']).counts(model) if job['variant'] != 'none' else dict(adapter_parameters=0)
        from ..table1_frozen.diagnostics import quantization_record
        write_json(path / 'quantization.json', quantization_record(model.backbone if job['variant'] != 'none' else model))
        write_json(path / 'done.json', dict(config_hash=key, selected_epoch=best['epoch'], validation=validation,
                   initial_backbone_sha256=initial, state_sha256=digest(model),
                   checkpoint_file_sha256=sha(path / 'best.pt'), full_split=not study['smoke'],
                   training_seconds=sum(r['seconds'] for r in history), peak_cuda_memory_bytes=peak_memory,
                   device=str(device), **extra))
    finally:
        close(model)


def test_manifest(root, study):
    validate_study(study)
    if study['smoke']:
        raise RuntimeError('Smoke never evaluates test metrics')
    cases, pairs = [], {}
    for job in study['jobs']:
        path = root / job['directory']
        if not (path / 'done.json').exists():
            raise RuntimeError('All paired training jobs must complete before testing')
        done = read(path / 'done.json')
        if done['config_hash'] != job_key(study, job) or not done['full_split']:
            raise ValueError('Completed fit no longer matches registration')
        if sha(path / 'best.pt') != done['checkpoint_file_sha256']:
            raise ValueError('Selected checkpoint changed')
        pair = pairs.setdefault((job['cell'], job['seed']), {})
        if job['variant'] in pair:
            raise ValueError('Duplicate paired variant')
        pair[job['variant']] = (done['initial_backbone_sha256'], read(path / 'split.json'))
        cases.append(dict(cell=job['cell'], seed=job['seed'], method=job['variant'], directory=job['directory'],
                          checkpoint_file_sha256=done['checkpoint_file_sha256'], state_sha256=done['state_sha256']))
    for pair in pairs.values():
        adaptation = study.get('adaptation', 'graphlora')
        if set(pair) != set(study['variants']) or pair['none'] != pair[adaptation]:
            raise ValueError('Paired backbone initialization or data split differs')
    return dict(cases=cases, sha256=config_hash(cases))


def finish(root, study, device):
    from ..table1_frozen.finish import test_once
    with open(root / '.final.lock', 'a') as guard:
        fcntl.flock(guard, fcntl.LOCK_EX)
        manifest = test_manifest(root, study)
        lock = root / 'TEST_LOCK.json'
        if lock.exists() and read(lock) != manifest:
            raise ValueError('Test lock changed')
        write_json(lock, manifest)
        for job, case in zip(study['jobs'], manifest['cases']):
            path = root / job['directory']
            if (path / 'test.json').exists():
                previous = read(path / 'test.json')
                if (previous['checkpoint_file_sha256'] != case['checkpoint_file_sha256'] or
                    previous['test_lock_sha256'] != manifest['sha256']):
                    raise ValueError('Existing test does not match registration')
                continue
            bundle = Bundle(job['config'])
            model, _ = model_for(study, job, bundle, device)
            try:
                load_state(model, torch.load(path / 'best.pt', map_location='cpu', weights_only=False)['model'])
                model.eval()
                if digest(model) != case['state_sha256']:
                    raise ValueError('Loaded checkpoint differs')
                test_once(root, case, model, job['config'], bundle, device)
            finally:
                close(model)
        write_json(root / 'completion_audit.json', dict(passed=True, fits=len(study['jobs']),
                   tests=len(study['jobs']), test_evaluations_per_model=1, paired_initializations_and_splits=True))


def generate(root, study):
    fits = sum((root / j['directory'] / 'done.json').exists() for j in study['jobs'])
    tests = sum((root / j['directory'] / 'test.json').exists() for j in study['jobs'])
    failures = [j['directory'] for j in study['jobs'] if (root / j['directory'] / 'failure.json').exists()]
    write_json(root / 'status.json', dict(fits=fits, fits_total=len(study['jobs']), tests=tests,
        tests_expected=0 if study['smoke'] else len(study['jobs']), failures=failures,
        complete=(root / 'completion_audit.json').exists()))
    rows = []
    for cell in study['cells']:
        for variant in study['variants']:
            selected = [j for j in study['jobs'] if j['cell'] == cell and j['variant'] == variant]
            scores = [read(root / j['directory'] / 'test.json')['metric'] for j in selected
                      if (root / j['directory'] / 'test.json').exists()]
            vals = [read(root / j['directory'] / 'done.json')['validation']['metric'] for j in selected
                    if (root / j['directory'] / 'done.json').exists()]
            rows.append(dict(cell=cell, variant=variant, fits=len(vals), tests=len(scores),
                             validation_mean=statistics.mean(vals) if vals else None,
                             test_mean=statistics.mean(scores) if scores else None,
                             test_sd=statistics.stdev(scores) if len(scores) > 1 else None))
    write_json(root / 'summary.json', rows)
    stream = io.StringIO(); writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader(); writer.writerows(rows)
    (root / 'summary.csv').write_text(stream.getvalue())


def worker(root, device):
    root = Path(root).resolve()
    if torch.device(device).type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('Official paired experiments and real-data smoke require CUDA')
    torch.cuda.set_device(device); torch.set_num_threads(4)
    verify_sources(root)
    study = read(root / 'study.json'); validate_study(study)
    for job in study['jobs']:
        if (root / 'STOP').exists():
            return
        path = root / job['directory']
        with open(path / '.fit.lock', 'a') as guard:
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if (path / 'done.json').exists() or (path / 'failure.json').exists():
                continue
            try:
                fit(root, study, job, device)
            except Exception:
                write_json(path / 'failure.json', dict(traceback=traceback.format_exc(), device=device))
                raise
            finally:
                generate(root, study)
        torch.cuda.empty_cache()
    if all((root / j['directory'] / 'done.json').exists() for j in study['jobs']):
        if study['smoke']:
            write_json(root / 'completion_audit.json', dict(passed=True, fits=len(study['jobs']), tests=0, smoke=True))
        else:
            finish(root, study, device)
        generate(root, study)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['prepare', 'worker', 'report'])
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--data-root', type=Path, default=Path('data'))
    parser.add_argument('--profile', choices=PROFILES, default='cora')
    parser.add_argument('--cells', nargs='+')
    parser.add_argument('--seeds', nargs='+', type=int)
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--normalization-floor', type=float, default=.1)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    if args.action == 'prepare':
        prepare(args.root, args.data_root, args.cells or PROFILES[args.profile], args.seeds, args.smoke, args.normalization_floor)
    elif args.action == 'worker':
        worker(args.root, args.device)
    else:
        root = args.root.resolve(); generate(root, read(root / 'study.json'))


if __name__ == '__main__':
    main()
