"""Bounded full-data successive halving, resume, and validation-only selection."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import shutil
import time
import traceback

import torch

from ..table1_frozen.backends import build, load_state, project
from ..table1_frozen.common import (config_hash, consider, digest, finite_gradients, file_lock,
                                   restore_rng, rng_state, save, seed_all, write_json)
from ..table1_frozen.data import Bundle
from ..table1_frozen.diagnostics import quantization_record
from ..table1_frozen.train import evaluate
from ..table1_gpf_rerun.model import GPFLoRAP, counts, make_optimizer
from ..table1_gpf_rerun.run import read, sha, snapshot, verify_sources, train_epoch
from .space import CELLS, SEEDS, candidates, instantiate

REPO = Path(__file__).resolve().parents[3]


def prepare(root, previous, smoke=False):
    root, previous = Path(root).resolve(), Path(previous).resolve()
    if root.exists():
        raise FileExistsError('Use a new output directory; worker resumes an existing study')
    prior, audit = read(previous / 'study.json'), read(previous / 'completion_audit.json')
    if not audit['passed'] or prior['seeds'] != SEEDS:
        raise ValueError('Expected completed three-seed GPF rerun reference')
    old_jobs = {(j['cell'], j['seed']): j for j in prior['jobs']}
    options = candidates()
    jobs = []
    for cell in CELLS:
        for candidate in options:
            for seed in SEEDS:
                original = old_jobs[(cell, seed)]
                path = previous / original['directory']
                done, test = read(path / 'done.json'), read(path / 'test.json')
                if sha(path / 'best.pt') != done['checkpoint_file_sha256'] or test['completed_test_evaluations'] != 1:
                    raise ValueError('Previous GPF evidence changed')
                c, hp = instantiate(candidate, original['config'])
                jobs.append(dict(cell=cell, candidate=candidate['id'], seed=seed, config=c,
                    hyperparameters=hp, epochs=2 if smoke else original['epochs'],
                    screen_epochs=1 if smoke else (original['epochs'] if c['node'] else 30),
                    directory=f"{cell}/candidate_{candidate['id']}/seed_{seed}",
                    reference=original['reference'], graphlora_reference=original['graphlora_reference'],
                    previous_gpf=dict(directory=str(path), done=done, test=test)))
    study = dict(version=1, method='GPF-LoRAP validation-tuned joint', cells=CELLS, seeds=SEEDS,
        search_seed=41, candidates=options, promote=2, jobs=jobs, smoke=smoke,
        subset_limit=8 if smoke else None, protocol='joint',
        selection='lowest validation task loss; ties by candidate ID; checkpoints by earliest epoch',
        scope='six cells selected after prior reported tests; exploratory tuning, not an untouched holdout',
        final_tests=len(CELLS) * len(SEEDS), previous=str(previous), created_unix=time.time())
    study['registration_sha256'] = config_hash(study)
    root.mkdir(parents=True)
    write_json(root / 'study.json', study)
    for job in jobs:
        (root / job['directory']).mkdir(parents=True)
    snapshot(root)
    # Include this protocol document in the immutable source snapshot too.
    manifest = read(root / 'source_manifest.json')
    for source in Path(__file__).parent.glob('*.md'):
        relative = source.relative_to(REPO)
        target = root / 'source_snapshot' / relative
        shutil.copy2(source, target)
        manifest[str(relative)] = sha(target)
    write_json(root / 'source_manifest.json', manifest)
    from .report import generate
    generate(root)
    return study


def check_registration(study):
    content = {k: v for k, v in study.items() if k != 'registration_sha256'}
    if config_hash(content) != study['registration_sha256']:
        raise ValueError('Registered search settings changed')


def fit_config(study, job):
    return dict(job=job, subset_limit=study['subset_limit'], protocol=study['protocol'])


def create_model(study, job, bundle, device):
    c, hp, seed = job['config'], job['hyperparameters'], job['seed']
    seed_all(seed)
    base = build(c, bundle, seed).to(device)
    initial = digest(base)
    if initial != job['reference']['done']['initial_sha256']:
        raise AssertionError('Common backbone initialization differs')
    batch = next(bundle.batches('train', seed, 0, device, limit=2))
    model = GPFLoRAP(base, c, seed, batch, hp['rank'], hp['num_prompts'],
        input_scale=hp['input_scale'], aggregation_scale=hp['aggregation_scale'], zero_init=hp['zero_init'])
    if digest(base) != initial:
        raise AssertionError('Prompt construction changed backbone')
    with torch.cuda.device(device):
        torch.cuda.manual_seed(seed + hp['training_cuda_rng_offset'])
    return model, initial


def checked_record(root, study, job, filename):
    path = root / job['directory'] / filename
    if not path.exists():
        return None
    record = read(path)
    if record['config_hash'] != config_hash(fit_config(study, job)):
        raise ValueError(f'Saved record configuration changed: {path}')
    return record


def fit(root, study, job, device, until):
    """The cap is execution control, not part of optimizer/scheduler configuration."""
    path = root / job['directory']
    config = fit_config(study, job)
    key = config_hash(config)
    if not 1 <= until <= job['epochs']:
        raise ValueError('Invalid epoch cap')
    if checked_record(root, study, job, 'done.json'):
        return
    if (path / 'config.json').exists() and read(path / 'config.json') != config:
        raise ValueError('Interrupted fit configuration changed')
    write_json(path / 'config.json', config)
    c, hp, seed = job['config'], job['hyperparameters'], job['seed']
    bundle = Bundle(c)
    write_json(path / 'split.json', bundle.manifest())
    model, initial = create_model(study, job, bundle, device)
    opt = make_optimizer(model, c, prompt_lr=hp['prompt_lr'], prompt_weight_decay=hp['prompt_weight_decay'])
    scheduler = (torch.optim.lr_scheduler.ReduceLROnPlateau(opt, factor=.5, patience=5)
        if c['framework'] == 'A2Q' and not c['node'] else
        torch.optim.lr_scheduler.StepLR(opt, 50, .5) if c['framework'] != 'A2Q' else None)
    history, best, start_epoch, peak = [], None, 1, 0
    try:
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
            write_json(path / 'initialization.json', dict(common_backbone_sha256=initial, **counts(model)))
        torch.cuda.reset_peak_memory_stats(device)
        def record_screen():
            shutil.copy2(path / 'best.pt', path / 'screen_best.pt')
            selected = history[best['epoch'] - 1]['validation']
            write_json(path / 'screen.json', dict(config_hash=key, epochs=job['screen_epochs'],
                validation=selected, selected_epoch=best['epoch'],
                checkpoint_file_sha256=sha(path / 'screen_best.pt'),
                training_seconds=sum(r['seconds'] for r in history), peak_cuda_memory_bytes=peak,
                device=device, **counts(model)))
        for epoch in range(start_epoch, until + 1):
            if (root / 'STOP').exists():
                return
            started = time.monotonic()
            train = train_epoch(model, job, dict(study, hyperparameters=hp), bundle, device, epoch, opt)
            val = evaluate(model, c, bundle, device, seed, limit=study['subset_limit'])
            previous = best
            best = consider(best, val['loss'], epoch, model)
            if previous is not best:
                save(path / 'best.pt', best)
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val['loss'])
            elif scheduler:
                scheduler.step()
            history.append(dict(epoch=epoch, train=train, validation=val, seconds=time.monotonic() - started))
            peak = max(peak, torch.cuda.max_memory_allocated(device))
            save(path / 'last.pt', dict(config_hash=key, epoch=epoch, model=model.state_dict(),
                optimizer=opt.state_dict(), scheduler=scheduler.state_dict() if scheduler else None,
                rng=rng_state(device), best=best, history=history, peak_cuda_memory_bytes=peak))
            write_json(path / 'history.json', history)
            write_json(path / 'progress.json', dict(epoch=epoch, epochs=job['epochs'], cap=until,
                best_epoch=best['epoch'], validation=val, device=device))
            if epoch == 1 or epoch % 10 == 0:
                from .report import generate
                generate(root)
                print(f"{job['cell']} {job['candidate']} s{seed} {epoch}/{until} val={val['metric']:.5f}", flush=True)
            if epoch == job['screen_epochs'] and seed == study['search_seed']:
                record_screen()
        # Recover a crash after the epoch checkpoint but before screen metadata.
        if seed == study['search_seed'] and not (path / 'screen.json').exists() and len(history) == job['screen_epochs']:
            record_screen()
        if until < job['epochs']:
            return
        load_state(model, best['model'])
        val = evaluate(model, c, bundle, device, seed, limit=study['subset_limit'])
        if abs(val['loss'] - best['validation_loss']) > 1e-5:
            raise AssertionError('Selected validation replay differs')
        write_json(path / 'quantization.json', quantization_record(model.backbone))
        write_json(path / 'done.json', dict(cell=job['cell'], candidate=job['candidate'], seed=seed,
            config_hash=key, selected_epoch=best['epoch'], validation=val, test_evaluations=0,
            full_split=not study['smoke'], common_backbone_sha256=initial, state_sha256=digest(model),
            checkpoint_file_sha256=sha(path / 'best.pt'), training_seconds=sum(r['seconds'] for r in history),
            peak_cuda_memory_bytes=peak, device=device, gpu_name=torch.cuda.get_device_name(device), **counts(model)))
    finally:
        model.close()


def _immutable_json(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Locked validation decision changed: {path}')
    else:
        write_json(path, value)


def tasks(root, study):
    """Only completed validation records influence promotion and selection."""
    pending, selections = [], {}
    with file_lock(root / '.plan.lock'):
        for cell in study['cells']:
            first = [j for j in study['jobs'] if j['cell'] == cell and j['seed'] == study['search_seed']]
            screens, rejected = [], []
            for job in first:
                record = checked_record(root, study, job, 'screen.json')
                if record is None:
                    failure = root / job['directory'] / 'failure.json'
                    if failure.exists() and read(failure).get('numerical_rejection'):
                        checked_record(root, study, job, 'failure.json')
                        rejected.append(job['candidate'])
                    else:
                        pending.append((job, job['screen_epochs']))
                else:
                    screens.append(dict(candidate=job['candidate'], validation_loss=record['validation']['loss']))
            if len(screens) + len(rejected) != len(first):
                continue
            if not screens:
                raise RuntimeError(f'No numerically valid candidate for {cell}')
            ranked = sorted(screens, key=lambda x: (x['validation_loss'], x['candidate']))
            promoted = [r['candidate'] for r in ranked[:study['promote']]]
            decision = dict(candidates=promoted, ranking=ranked, rejected=rejected,
                            criterion='validation_loss', seed=study['search_seed'])
            _immutable_json(root / cell / 'promotion.json', decision)
            full, full_rejected = [], []
            for job in first:
                if job['candidate'] not in promoted:
                    continue
                record = checked_record(root, study, job, 'done.json')
                if record is None:
                    failure = root / job['directory'] / 'failure.json'
                    if failure.exists() and read(failure).get('numerical_rejection'):
                        checked_record(root, study, job, 'failure.json')
                        full_rejected.append(job['candidate'])
                    else:
                        pending.append((job, job['epochs']))
                else:
                    full.append(dict(candidate=job['candidate'], validation_loss=record['validation']['loss']))
            if len(full) + len(full_rejected) != len(promoted):
                continue
            if not full:
                raise RuntimeError(f'All promoted candidates failed for {cell}')
            final_rank = sorted(full, key=lambda x: (x['validation_loss'], x['candidate']))
            selected = dict(final_rank[0], ranking=final_rank, rejected=full_rejected,
                            criterion='validation_loss', seed=study['search_seed'])
            _immutable_json(root / cell / 'selection.json', selected)
            selections[cell] = selected
            for job in study['jobs']:
                if job['cell'] == cell and job['candidate'] == selected['candidate']:
                    if checked_record(root, study, job, 'done.json') is None:
                        pending.append((job, job['epochs']))
        if len(selections) == len(study['cells']):
            _immutable_json(root / 'SELECTION_LOCK.json', dict(cells=selections, sha256=config_hash(selections)))
    return pending


def selected_jobs(root, study):
    lockpath = root / 'SELECTION_LOCK.json'
    if not lockpath.exists():
        raise RuntimeError('All validation selections must be locked before testing')
    lock = read(lockpath)
    if lock['sha256'] != config_hash(lock['cells']) or set(lock['cells']) != set(study['cells']):
        raise ValueError('Invalid selection lock')
    for cell, decision in lock['cells'].items():
        if read(root / cell / 'selection.json') != decision:
            raise ValueError('Selected configuration changed')
    jobs = [j for j in study['jobs'] if j['candidate'] == lock['cells'][j['cell']]['candidate']]
    for cell in study['cells']:
        actual = [j['seed'] for j in jobs if j['cell'] == cell]
        if sorted(actual) != sorted(study['seeds']):
            raise ValueError('Selected candidate must cover exactly the registered seeds')
    return jobs


def lock_tests(root, study):
    if study['smoke']:
        raise RuntimeError('Smoke studies never access tests')
    check_registration(study)
    jobs = selected_jobs(root, study)
    cases = []
    for job in jobs:
        done = checked_record(root, study, job, 'done.json')
        if done is None:
            raise RuntimeError('All selected three-seed fits must finish before testing')
        if not done['full_split'] or done['test_evaluations'] != 0:
            raise ValueError('Invalid formal fit')
        if sha(root / job['directory'] / 'best.pt') != done['checkpoint_file_sha256']:
            raise ValueError('Selected checkpoint changed')
        cases.append(dict(cell=job['cell'], seed=job['seed'], method='gpf_lorap_tuned',
            directory=job['directory'], state_sha256=done['state_sha256'],
            checkpoint_file_sha256=done['checkpoint_file_sha256']))
    manifest = dict(cases=cases, sha256=config_hash(cases), selection_sha256=read(root / 'SELECTION_LOCK.json')['sha256'])
    _immutable_json(root / 'TEST_LOCK.json', manifest)
    return jobs, manifest


def finish(root, study, device):
    from ..table1_frozen.finish import test_once
    with open(root / '.test.lock', 'a') as guard:
        try:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return
        jobs, manifest = lock_tests(root, study)
        for job, case in zip(jobs, manifest['cases']):
            path = root / job['directory']
            if (path / 'test.json').exists():
                result = read(path / 'test.json')
                if (result['checkpoint_file_sha256'] != case['checkpoint_file_sha256'] or
                    result['test_lock_sha256'] != manifest['sha256'] or result.get('completed_test_evaluations') != 1):
                    raise ValueError('Existing test belongs to another selected model')
                continue
            bundle = Bundle(job['config'])
            model, _ = create_model(study, job, bundle, device)
            try:
                load_state(model, torch.load(path / 'best.pt', map_location='cpu', weights_only=False)['model'])
                model.eval()
                if digest(model) != case['state_sha256']:
                    raise ValueError('Loaded selected state differs')
                test_once(root, case, model, job['config'], bundle, device)
            finally:
                model.close()
        write_json(root / 'completion_audit.json', dict(passed=True, tests=len(jobs),
            test_evaluations_per_model=1, seeds=study['seeds'], cells=len(study['cells'])))
        from .report import generate
        generate(root)


def worker(root, device):
    root = Path(root).resolve()
    if not torch.cuda.is_available() or torch.device(device).type != 'cuda':
        raise RuntimeError('CUDA required')
    torch.cuda.set_device(device)
    torch.set_num_threads(4)
    verify_sources(root)
    if not Path(__file__).resolve().is_relative_to(root / 'source_snapshot'):
        raise RuntimeError('Launch the worker with its source_snapshot as working directory')
    study = read(root / 'study.json')
    check_registration(study)
    from .report import generate
    while not (root / 'STOP').exists():
        pending = tasks(root, study)
        if not pending:
            if study['smoke']:
                write_json(root / 'completion_audit.json', dict(passed=True, smoke=True, tests=0,
                    selected_fits=len(selected_jobs(root, study))))
                generate(root)
            else:
                finish(root, study, device)
            return
        worked = False
        for job, cap in pending:
            path = root / job['directory']
            if (path / 'failure.json').exists():
                continue
            with open(path / '.fit.lock', 'a') as guard:
                try:
                    fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    continue
                marker = 'done.json' if cap == job['epochs'] else 'screen.json'
                if checked_record(root, study, job, marker):
                    continue
                write_json(path / 'worker.json', dict(pid=os.getpid(), device=device,
                    started_unix=time.time(), cap=cap, candidate=job['candidate']))
                generate(root)
                try:
                    fit(root, study, job, device, cap)
                except Exception as error:
                    rejected = isinstance(error, FloatingPointError) and job['seed'] == study['search_seed']
                    write_json(path / 'failure.json', dict(traceback=traceback.format_exc(), device=device,
                        config_hash=config_hash(fit_config(study, job)), cap=cap, numerical_rejection=rejected))
                    if not rejected:
                        raise
                finally:
                    (path / 'worker.json').unlink(missing_ok=True)
                    generate(root)
                worked = True
            torch.cuda.empty_cache()
            break  # Replan to release promotions and confirmations promptly.
        if not worked:
            # Other workers hold queued jobs; avoid busy polling. A failed job
            # remains visible and stops automatic phase advancement.
            if all((root / j['directory'] / 'failure.json').exists() for j, _ in pending):
                raise RuntimeError('All pending jobs failed; inspect failure.json')
            time.sleep(10)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('prepare', 'worker', 'finish', 'report'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--previous', default=str(REPO / 'result/lorap_study/table1_gpf_rerun_3seeds_20261001_v1'))
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if args.action == 'prepare':
        prepare(root, args.previous, args.smoke)
    elif args.action == 'worker':
        worker(root, args.device)
    elif args.action == 'finish':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA required')
        torch.cuda.set_device(args.device)
        torch.set_num_threads(4)
        verify_sources(root)
        finish(root, read(root / 'study.json'), args.device)
    else:
        from .report import generate
        generate(root)


if __name__ == '__main__':
    main()
