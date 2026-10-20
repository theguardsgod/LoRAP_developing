"""Registered refinement rounds with protected incumbents and three-seed selection."""
from pathlib import Path
import shutil
import statistics
import time

from ..table1_frozen.common import config_hash, file_lock, write_json
from . import run
from .refine_space import CELLS, candidates


def prepare(root, previous, smoke=False, round_number=2):
    root, previous = Path(root).resolve(), Path(previous).resolve()
    if root.exists():
        raise FileExistsError('Choose a new output directory')
    if round_number == 2:
        cells, make_candidates, screen_epochs, promoted_new = CELLS, candidates, 60, 2
    elif round_number == 3:
        from .refine3_space import CELLS as cells, candidates as make_candidates
        screen_epochs, promoted_new = 80, 3
    elif round_number == 4:
        from .refine4_space import CELLS as cells, candidates as make_candidates
        screen_epochs, promoted_new = 80, 4
    else:
        raise ValueError('Supported refinement rounds: 2, 3, 4')
    prior = run.read(previous / 'study.json')
    run.check_registration(prior)
    if not run.read(previous / 'completion_audit.json')['passed'] or prior['seeds'] != [41, 42, 43]:
        raise ValueError('A completed three-seed prior search is required')
    if prior['smoke']:
        raise ValueError('Incumbents must come from a formal completed study')
    selection_lock = run.read(previous / 'SELECTION_LOCK.json')
    selected = selection_lock['cells']
    if selection_lock['sha256'] != config_hash(selected):
        raise ValueError('Prior selection lock changed')
    jobs, options, reusable = [], {}, []
    for cell in cells:
        old = {j['seed']: j for j in prior['jobs'] if j['cell'] == cell and j['candidate'] == selected[cell]['candidate']}
        if sorted(old) != [41, 42, 43]:
            raise ValueError('Incomplete incumbent seed coverage')
        candidates_here = make_candidates(cell, old[41]['config'], old[41]['hyperparameters'])
        if round_number >= 3 and (candidates_here[0]['config'] != old[41]['config'] or
                                  candidates_here[0]['hp'] != old[41]['hyperparameters']):
            raise ValueError('Reused incumbent settings must be preserved exactly')
        options[cell] = candidates_here
        for candidate in candidates_here:
            for seed in prior['seeds']:
                reference = old[seed]
                path = previous / reference['directory']
                done, test = run.read(path / 'done.json'), run.read(path / 'test.json')
                if done['config_hash'] != config_hash(run.fit_config(prior, reference)):
                    raise ValueError('Incumbent configuration changed')
                if run.sha(path / 'best.pt') != done['checkpoint_file_sha256'] or test['checkpoint_file_sha256'] != done['checkpoint_file_sha256']:
                    raise ValueError('Incumbent checkpoint changed')
                if test['completed_test_evaluations'] != 1:
                    raise ValueError('Incumbent test provenance invalid')
                epochs = 2 if smoke else candidate.get('epochs', reference['epochs'])
                screen = 1 if smoke else (epochs if candidate['id'] == 'r00' or candidate['config']['node'] else screen_epochs)
                job = dict(cell=cell, candidate=candidate['id'], seed=seed, config=candidate['config'],
                    hyperparameters=candidate['hp'], epochs=epochs, screen_epochs=screen,
                    directory=f"{cell}/candidate_{candidate['id']}/seed_{seed}",
                    reference=reference['reference'], graphlora_reference=reference['graphlora_reference'],
                    previous_gpf=dict(directory=str(path), done=done, test=test),
                    original_gpf=reference.get('original_gpf', reference['previous_gpf']))
                if candidate['id'] == 'r00' and not smoke:
                    job['reuse_from'] = dict(directory=str(path), checkpoint_file_sha256=done['checkpoint_file_sha256'],
                        source_config_hash=done['config_hash'], test_file_sha256=run.sha(path / 'test.json'))
                    reusable.append((job, path, done))
                jobs.append(job)
    study = dict(version=round_number, refinement_round=round_number,
        method=f'GPF-LoRAP validation refinement round {round_number}', selection_protocol='mean_three',
        cells=cells, seeds=[41, 42, 43], selection_seeds=[41, 42, 43], search_seed=41, jobs=jobs,
        candidates=[dict(id=c['id']) for c in options[cells[0]]], candidates_by_cell=options,
        promote=promoted_new + 1, promoted_new=promoted_new, screen_epochs=screen_epochs,
        protected_candidate='r00', smoke=smoke, subset_limit=8 if smoke else None,
        protocol='joint', previous=str(previous), created_unix=time.time(),
        selection='minimum mean of three per-seed best validation losses; ties by candidate ID',
        scope=f'remaining {len(cells)} cells after prior tests; exploratory validation-only new choices',
        final_tests=len(cells) * 3, prior_tests_may_be_reused=True)
    if round_number == 4:
        study['comparison_note'] = ('本轮扩大训练超参数搜索：分类头 dropout、BatchNorm momentum、batch size、'
            'Adam/AdamW、学习率/调度器、权重衰减及提示容量。新候选完整训练 300 轮，'
            '旧配置和固定 GraphLoRA 参照为 200 轮。网络宽度、层数、数据划分、4-bit 量化器规则'
            '及任务损失不变。属于扩大训练预算后的探索结果，不是相同训练设置下的单模块消融。')
    study['registration_sha256'] = config_hash(study)
    root.mkdir(parents=True)
    write_json(root / 'study.json', study)
    for job in jobs:
        (root / job['directory']).mkdir(parents=True)
    for job, source, done in reusable:
        target = root / job['directory']
        for name in ('best.pt', 'history.json', 'split.json', 'initialization.json', 'quantization.json'):
            shutil.copy2(source / name, target / name)
        key = config_hash(run.fit_config(study, job))
        write_json(target / 'config.json', run.fit_config(study, job))
        write_json(target / 'done.json', dict(done, candidate='r00', config_hash=key,
            reused_training=True, original_training_directory=str(source)))
        if job['seed'] == 41:
            shutil.copy2(source / 'best.pt', target / 'screen_best.pt')
            write_json(target / 'screen.json', dict(done, config_hash=key, epochs=job['epochs'],
                reused_training=True, protected_incumbent=True))
    run.snapshot(root)
    manifest = run.read(root / 'source_manifest.json')
    for source in Path(__file__).parent.glob('*.md'):
        relative = source.relative_to(run.REPO)
        target = root / 'source_snapshot' / relative
        shutil.copy2(source, target)
        manifest[str(relative)] = run.sha(target)
    write_json(root / 'source_manifest.json', manifest)
    from .refine_report import generate
    generate(root)
    return study


def tasks(root, study):
    pending, selections = [], {}
    with file_lock(root / '.plan.lock'):
        for cell in study['cells']:
            first = [j for j in study['jobs'] if j['cell'] == cell and j['seed'] == study['search_seed']]
            valid, rejected = [], []
            for job in first:
                screen = run.checked_record(root, study, job, 'screen.json')
                if screen is not None:
                    valid.append(dict(candidate=job['candidate'], validation_loss=screen['validation']['loss']))
                    continue
                failure = root / job['directory'] / 'failure.json'
                if failure.exists() and run.read(failure).get('numerical_rejection'):
                    run.checked_record(root, study, job, 'failure.json')
                    rejected.append(job['candidate'])
                else:
                    pending.append((job, job['screen_epochs']))
            if len(valid) + len(rejected) != len(first):
                continue
            incumbent = study['protected_candidate']
            if incumbent in rejected:
                raise RuntimeError('Protected incumbent failed')
            ranked = sorted([r for r in valid if r['candidate'] != incumbent],
                            key=lambda r: (r['validation_loss'], r['candidate']))
            promoted = [incumbent] + [r['candidate'] for r in ranked[:study['promoted_new']]]
            run._immutable_json(root / cell / 'promotion.json', dict(candidates=promoted,
                ranking=ranked, rejected=rejected, protected_candidate=incumbent,
                criterion='seed41 validation loss of new candidates; incumbent always retained'))
            full, invalid = {}, []
            cell_pending = []
            for candidate in promoted:
                records = []
                for job in [j for j in study['jobs'] if j['cell'] == cell and j['candidate'] == candidate]:
                    done = run.checked_record(root, study, job, 'done.json')
                    if done is not None:
                        records.append(dict(seed=job['seed'], loss=done['validation']['loss']))
                    else:
                        failure = root / job['directory'] / 'failure.json'
                        if failure.exists() and run.read(failure).get('numerical_rejection'):
                            run.checked_record(root, study, job, 'failure.json')
                            invalid.append(candidate)
                        else:
                            cell_pending.append((job, job['epochs']))
                if len(records) == len(study['selection_seeds']):
                    if sorted(r['seed'] for r in records) != sorted(study['selection_seeds']):
                        raise ValueError('Mean validation requires exactly the registered seeds')
                    full[candidate] = dict(candidate=candidate, validation_loss=statistics.mean(r['loss'] for r in records),
                                           per_seed_validation=records)
            # A numerical failure excludes the whole candidate, never individual seeds.
            cell_pending = [(j, cap) for j, cap in cell_pending if j['candidate'] not in invalid]
            pending.extend(cell_pending)
            if cell_pending or len(full) + len(set(invalid)) != len(promoted):
                continue
            if not full:
                raise RuntimeError('All promoted candidates failed')
            ranking = sorted(full.values(), key=lambda r: (r['validation_loss'], r['candidate']))
            choice = dict(ranking[0], ranking=ranking, rejected=sorted(set(invalid)),
                selection_seeds=study['selection_seeds'], criterion='mean three-seed validation loss')
            run._immutable_json(root / cell / 'selection.json', choice)
            selections[cell] = choice
        if len(selections) == len(study['cells']):
            run._immutable_json(root / 'SELECTION_LOCK.json', dict(cells=selections, sha256=config_hash(selections)))
    return pending
