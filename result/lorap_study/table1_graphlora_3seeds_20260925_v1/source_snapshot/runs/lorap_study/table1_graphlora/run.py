"""Isolated 3-seed Table I extension with archived controls and a test gate."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import torch
from ..table1_frozen.backends import build, load_state, predict, project
from ..table1_frozen.common import (config_hash, consider, digest, finite_gradients, restore_rng,
                                   rng_state, save, seed_all, write_json)
from ..table1_frozen.data import Bundle
from ..table1_frozen.diagnostics import quantization_record
from ..table1_frozen.registry import CELLS
from ..table1_frozen.train import evaluate
from .model import GraphLoRA, counts, make_optimizer, objective

DEFAULT_SEEDS = [41, 42, 43]
HP = dict(rank=2, contrast_weight=.1, structure_weight=.1, tau=.5, supervised_weight=.2)
REPO = Path(__file__).resolve().parents[3]


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def snapshot(root):
    files = list(REPO.glob('*.py'))
    for directory in ('runs/lorap_study', 'quantize_function', 'quantize_method', 'prompt', 'utils'):
        files.extend((REPO / directory).rglob('*.py'))
    files.extend((REPO / 'runs/lorap_study/table1_graphlora').glob('*.md'))
    files.extend(REPO / 'runs/lorap_study' / name for name in ('GRAPHLORA_METHOD.md',))
    files.extend((REPO / 'runs/lorap_study/graphlora_official').rglob('*.json'))
    manifest = {}
    for source in sorted(set(files)):
        relative = source.relative_to(REPO)
        target = root / 'source_snapshot' / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        manifest[str(relative)] = sha(target)
    write_json(root / 'source_manifest.json', manifest)


def verify_sources(root):
    for name, expected in read(root / 'source_manifest.json').items():
        if sha(root / 'source_snapshot' / name) != expected:
            raise ValueError(f'Archived source changed: {name}')


def prepare(root, reference, cells=None, seeds=None, protocol='joint', smoke=False):
    root, reference = Path(root).resolve(), Path(reference).resolve()
    if root.exists():
        raise FileExistsError('Choose a new output directory, or use worker/finish to resume')
    cells, seeds = list(CELLS) if cells is None else cells, DEFAULT_SEEDS if seeds is None else seeds
    if not seeds or len(seeds) != len(set(seeds)) or any(s not in DEFAULT_SEEDS for s in seeds):
        raise ValueError('Use unique seeds from the paired reference: 41, 42, 43')
    if not cells or len(cells) != len(set(cells)) or any(n not in CELLS for n in cells):
        raise ValueError('Choose unique registered Table I cells')
    if protocol not in ('joint', 'frozen'):
        raise ValueError('Unknown training protocol')
    jobs = []
    # Check every reference before creating an experiment. Existing baseline
    # tests are copied for reporting only, never used to choose hyperparameters.
    for name in cells:
        for seed in seeds:
            path = reference / name / f'seed_{seed}' / 'baseline'
            config, done, test = (read(path / f'{file}.json') for file in ('config', 'done', 'test'))
            if not done['full_split'] or test['completed_test_evaluations'] != 1:
                raise ValueError(f'Incomplete reference {path}')
            if sha(path / 'best.pt') != test['checkpoint_file_sha256']:
                raise ValueError(f'Reference checkpoint changed: {path}')
            if config_hash(config) != done['config_hash']:
                raise ValueError(f'Reference config changed: {path}')
            c = {k: v for k, v in config.items() if k not in ('seed', 'subset_limit', 'stage')}
            epochs = 2 if smoke else c['baseline_epochs'] if protocol == 'joint' else c['adapter_epochs']
            jobs.append(dict(cell=name, seed=seed, config=c, epochs=epochs,
                             directory=f'{name}/seed_{seed}',
                             reference=dict(directory=str(path), config=config, done=done, test=test,
                                            checkpoint_file_sha256=sha(path / 'best.pt'))))
    study = dict(version=1, method='GraphLoRA-adapted', protocol=protocol, seeds=seeds, cells=cells,
                 hyperparameters=HP, jobs=jobs, training_jobs=len(jobs),
                 smoke=smoke, subset_limit=8 if smoke else None,
                 test_policy='all registered fits complete before one test per selected model',
                 controls_reused=True, tuning_candidates_per_cell=1,
                 created_unix=time.time(), torch_version=torch.__version__)
    root.mkdir(parents=True)
    write_json(root / 'study.json', study)
    for job in jobs:
        write_json(root / job['directory'] / 'reference.json', job['reference'])
    snapshot(root)
    from .report import generate
    generate(root)
    return study


def create_model(study, job, bundle, device):
    c, seed = job['config'], job['seed']
    seed_all(seed)
    base = build(c, bundle, seed).to(device)
    initial = digest(base)
    if initial != job['reference']['done']['initial_sha256']:
        raise AssertionError(f"Common initialization changed: {job['cell']} seed={seed}")
    if study['protocol'] == 'frozen':
        path = Path(job['reference']['directory']) / 'best.pt'
        if sha(path) != job['reference']['checkpoint_file_sha256']:
            raise ValueError('Frozen reference changed')
        load_state(base, torch.load(path, map_location='cpu', weights_only=False)['model'])
    before = digest(base)
    example = next(bundle.batches('train', seed, 0, device, limit=2))
    model = GraphLoRA(base, c, seed, example, study['hyperparameters']['rank'], study['protocol'] == 'frozen')
    if digest(base) != before:
        raise AssertionError('Shape probe or adapter construction changed backbone state')
    return model, initial, before


def train_epoch(model, job, study, bundle, device, epoch, opt):
    model.train()
    totals = torch.zeros(6, device=device, dtype=torch.float64)
    count = 0
    for batch in bundle.batches('train', job['seed'], epoch, device, limit=study['subset_limit']):
        opt.zero_grad(set_to_none=True)
        loss, logits, y, parts = objective(model, job['config'], batch, study['hyperparameters'])
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite training objective')
        loss.backward()
        finite_gradients(model)
        opt.step()
        project(model, job['config'])
        metric = (logits.flatten() - y).abs().sum() if job['config']['regression'] else (logits.argmax(1) == y).sum()
        values = [parts[k].detach().double() * len(y) for k in ('task', 'contrast', 'structure', 'storage')]
        totals += torch.stack([*values, loss.detach().double() * len(y), metric.detach().double()])
        count += len(y)
    values = (totals / count).cpu().tolist()
    if not job['config']['regression']:
        values[-1] *= 100
    return dict(zip(('loss', 'contrast', 'structure', 'storage', 'objective', 'metric'), values)) | dict(examples=count)


def fit(root, study, job, device):
    path = root / job['directory']
    config = dict(job=job, protocol=study['protocol'], hp=study['hyperparameters'], subset_limit=study['subset_limit'])
    key = config_hash(config)
    if (path / 'done.json').exists():
        if read(path / 'done.json')['config_hash'] != key:
            raise ValueError('Completed fit configuration changed')
        return
    if (path / 'config.json').exists() and read(path / 'config.json') != config:
        raise ValueError('Interrupted fit configuration changed')
    write_json(path / 'config.json', config)
    c, seed = job['config'], job['seed']
    bundle = Bundle(c)
    write_json(path / 'split.json', bundle.manifest())
    model, initial, before = create_model(study, job, bundle, device)
    opt = make_optimizer(model, c)
    scheduler = None if model.frozen else (
        torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=5)
        if c['framework'] == 'A2Q' and not c['node'] else
        torch.optim.lr_scheduler.StepLR(opt, 50, .5) if c['framework'] != 'A2Q' else None)
    history, best, start_epoch, peak = [], None, 1, 0
    if (path / 'last.pt').exists():
        last = torch.load(path / 'last.pt', map_location='cpu', weights_only=False)
        if last['config_hash'] != key:
            raise ValueError('Resume configuration changed')
        load_state(model, last['model'])
        opt.load_state_dict(last['optimizer'])
        if scheduler:
            scheduler.load_state_dict(last['scheduler'])
        history, best, start_epoch = last['history'], last['best'], last['epoch'] + 1
        peak = last['peak_cuda_memory_bytes']
        restore_rng(last['rng'], device)
    else:
        write_json(path / 'initialization.json', dict(common_backbone_sha256=initial,
                   starting_backbone_sha256=before, **counts(model)))
        if model.frozen:
            val = evaluate(model, c, bundle, device, seed, limit=study['subset_limit'])
            best = consider(None, val['loss'], 0, model)
            save(path / 'best.pt', best)
    torch.cuda.reset_peak_memory_stats(device)
    for epoch in range(start_epoch, job['epochs'] + 1):
        if (root / 'STOP').exists():
            model.close()
            return
        start = time.monotonic()
        train = train_epoch(model, job, study, bundle, device, epoch, opt)
        val = evaluate(model, c, bundle, device, seed, limit=study['subset_limit'])
        old = best
        best = consider(best, val['loss'], epoch, model)
        if old is not best:
            save(path / 'best.pt', best)
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val['loss'])
        elif scheduler:
            scheduler.step()
        if model.frozen and digest(model.backbone) != before:
            raise AssertionError('Frozen backbone changed')
        history.append(dict(epoch=epoch, train=train, validation=val, seconds=time.monotonic() - start))
        peak = max(peak, torch.cuda.max_memory_allocated(device))
        save(path / 'last.pt', dict(config_hash=key, epoch=epoch, model=model.state_dict(),
             optimizer=opt.state_dict(), scheduler=scheduler.state_dict() if scheduler else None,
             rng=rng_state(device), best=best, history=history, peak_cuda_memory_bytes=peak))
        write_json(path / 'history.json', history)
        write_json(path / 'progress.json', dict(epoch=epoch, epochs=job['epochs'], best_epoch=best['epoch'],
                                               validation=val, device=device))
        if epoch == 1 or epoch % 10 == 0:
            from .report import generate
            generate(root)
        if epoch == 1 or epoch % 25 == 0:
            print(f"{job['cell']} s{seed} {epoch}/{job['epochs']} val={val['metric']:.4f}", flush=True)
        if model.frozen and epoch >= c['min_epochs'] and epoch - best['epoch'] >= c['stop_patience']:
            break
    load_state(model, best['model'])
    val = evaluate(model, c, bundle, device, seed, limit=study['subset_limit'])
    if abs(val['loss'] - best['validation_loss']) > 1e-5:
        raise AssertionError('Selected checkpoint validation replay differs')
    write_json(path / 'quantization.json', quantization_record(model.backbone))
    write_json(path / 'done.json', dict(cell=job['cell'], seed=seed, config_hash=key,
               selected_epoch=best['epoch'], validation=val, test_evaluations=0,
               full_split=not study['smoke'], common_backbone_sha256=initial,
               state_sha256=digest(model), checkpoint_file_sha256=sha(path / 'best.pt'),
               training_seconds=sum(row['seconds'] for row in history), peak_cuda_memory_bytes=peak,
               device=device, gpu_name=torch.cuda.get_device_name(device), **counts(model)))
    model.close()


def lock_tests(root, study):
    if study['smoke']:
        raise RuntimeError('Smoke checks never access benchmark test sets')
    cases = []
    for job in study['jobs']:
        path = root / job['directory']
        if not (path / 'done.json').exists():
            raise RuntimeError('All registered training runs must finish before testing')
        row = read(path / 'done.json')
        if not row['full_split'] or row['test_evaluations'] != 0:
            raise ValueError('Invalid formal fit')
        expected = dict(job=job, protocol=study['protocol'], hp=study['hyperparameters'], subset_limit=None)
        if row['config_hash'] != config_hash(expected):
            raise ValueError('Fit no longer matches registered configuration')
        if sha(path / 'best.pt') != row['checkpoint_file_sha256']:
            raise ValueError('Selected checkpoint changed')
        cases.append(dict(cell=job['cell'], seed=job['seed'], method='graphlora', directory=job['directory'],
                          checkpoint_file_sha256=row['checkpoint_file_sha256'], state_sha256=row['state_sha256']))
    manifest = dict(cases=cases, sha256=config_hash(cases))
    if (root / 'TEST_LOCK.json').exists() and read(root / 'TEST_LOCK.json') != manifest:
        raise ValueError('Test lock changed')
    write_json(root / 'TEST_LOCK.json', manifest)
    return manifest


def finish(root, study, device):
    # Covers both automatic workers and the explicit finish entrypoint.
    with open(root / '.test.lock', 'a') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        return _finish_locked(root, study, device)


def _finish_locked(root, study, device):
    from ..table1_frozen.finish import test_once
    manifest = lock_tests(root, study)
    for job, case in zip(study['jobs'], manifest['cases']):
        path = root / case['directory']
        if (path / 'test.json').exists():
            result = read(path / 'test.json')
            if result['checkpoint_file_sha256'] != case['checkpoint_file_sha256'] or result['test_lock_sha256'] != manifest['sha256']:
                raise ValueError('Existing test result does not match this checkpoint')
            continue
        bundle = Bundle(job['config'])
        model, _, _ = create_model(study, job, bundle, device)
        load_state(model, torch.load(path / 'best.pt', map_location='cpu', weights_only=False)['model'])
        model.eval()
        if digest(model) != case['state_sha256']:
            raise ValueError('Loaded checkpoint state differs')
        test_once(root, case, model, job['config'], bundle, device)
        model.close()
    write_json(root / 'completion_audit.json', dict(passed=True, fits=len(study['jobs']),
               tests=len(study['jobs']), test_evaluations_per_model=1, seeds=study['seeds']))
    from .report import generate
    generate(root)


def worker(root, device):
    root = Path(root).resolve()
    if not torch.cuda.is_available() or torch.device(device).type != 'cuda':
        raise RuntimeError('Formal experiments and real-data smoke checks require CUDA')
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    verify_sources(root)
    study = read(root / 'study.json')
    from .report import generate
    for job in study['jobs']:
        if (root / 'STOP').exists():
            break
        path = root / job['directory']
        if (path / 'done.json').exists() or (path / 'failure.json').exists():
            continue
        with open(path / '.fit.lock', 'a') as guard:
            try:
                fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                continue
            if (path / 'done.json').exists():
                continue
            write_json(path / 'worker.json', dict(pid=os.getpid(), device=device, started_unix=time.time()))
            generate(root)
            try:
                fit(root, study, job, device)
            except Exception:
                write_json(path / 'failure.json', dict(traceback=traceback.format_exc(), device=device))
                generate(root)
                raise
            finally:
                (path / 'worker.json').unlink(missing_ok=True)
            generate(root)
        torch.cuda.empty_cache()
    if all((root / j['directory'] / 'done.json').exists() for j in study['jobs']):
        if study['smoke']:
            write_json(root / 'completion_audit.json', dict(passed=True, fits=len(study['jobs']), tests=0, smoke=True))
            generate(root)
        else:
            finish(root, study, device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=('prepare', 'worker', 'finish', 'report'))
    p.add_argument('--root', required=True)
    p.add_argument('--reference', default=str(REPO / 'result/lorap_study/table1_frozen_20260914_v2'))
    p.add_argument('--seeds', nargs='+', type=int, default=DEFAULT_SEEDS)
    p.add_argument('--cells', nargs='+')
    p.add_argument('--protocol', choices=('joint', 'frozen'), default='joint')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    root = Path(args.root).resolve()
    if args.action == 'prepare':
        prepare(root, args.reference, args.cells, args.seeds, args.protocol, args.smoke)
    elif args.action == 'worker':
        worker(root, args.device)
    elif args.action == 'finish':
        verify_sources(root)
        finish(root, read(root / 'study.json'), args.device)
    else:
        from .report import generate
        generate(root)


if __name__ == '__main__':
    main()
