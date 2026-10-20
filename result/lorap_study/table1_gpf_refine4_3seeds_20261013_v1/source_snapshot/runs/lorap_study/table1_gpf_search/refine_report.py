"""Saved-record reporting for refinement rounds, including cached incumbents."""
import csv
import io
from pathlib import Path

from ..table1_frozen.common import file_lock, write_json
from ..table1_graphlora.report import formatted, stats
from .report import read, _write_text


def generate(root):
    root = Path(root)
    with file_lock(root / '.report.lock'):
        return _generate(root)


def _generate(root):
    study = read(root / 'study.json')
    rows, trials, resources, pairs, running, failures = [], [], [], [], [], []
    screened = cached = validation_fits = tests = selected_cells = new_tests = 0
    for cell in study['cells']:
        cell_jobs = [j for j in study['jobs'] if j['cell'] == cell]
        promotion = read(root / cell / 'promotion.json', {})
        promoted = promotion.get('candidates', [])
        selection = read(root / cell / 'selection.json', {})
        chosen = selection.get('candidate')
        selected_cells += chosen is not None
        completed = cell_screened = 0
        values, graphs, olds, controls, gains = [], [], [], [], []
        references = {}
        for job in cell_jobs:
            path = root / job['directory']
            candidate, seed = job['candidate'], job['seed']
            references.setdefault(seed, job)
            screen, done, test = (read(path / name) for name in ('screen.json', 'done.json', 'test.json'))
            worker = read(path / 'worker.json')
            if worker:
                running.append(dict(worker, **read(path / 'progress.json', {}), cell=cell, seed=seed,
                                    directory=job['directory']))
            failure = read(path / 'failure.json')
            if failure:
                failures.append(dict(directory=job['directory'], numerical_rejection=failure.get('numerical_rejection', False)))
            if screen and candidate != study['protected_candidate'] and seed == study['search_seed']:
                screened += 1
                cell_screened += 1
            if done and job.get('reuse_from'):
                cached += 1
            if done and candidate in promoted:
                validation_fits += 1
                completed += 1
            record = done or screen
            if record:
                resources.append(dict(cell=cell, candidate=candidate, seed=seed,
                    reused_training=bool(job.get('reuse_from')), stage='full' if done else 'screen',
                    registered_epochs=job['epochs'], batch_size=job['config']['batch_size'],
                    head_dropout=job['config'].get('head_dropout', .5),
                    batchnorm_momentum=job['config'].get('batchnorm_momentum', .1),
                    optimizer=job['hyperparameters'].get('optimizer', 'adam'),
                    prompt_parameters=record.get('prompt_parameters'), total_parameters=record.get('total_parameters'),
                    prompt_KiB=record.get('adapter_fp32_bytes', 0) / 1024,
                    training_seconds=record.get('training_seconds'),
                    peak_cuda_MiB=record.get('peak_cuda_memory_bytes', 0) / 1024**2, device=record.get('device')))
            trials.append(dict(cell=cell, candidate=candidate, seed=seed, promoted=candidate in promoted,
                selected=candidate == chosen, screened=bool(screen), complete=bool(done),
                screen_loss=None if not screen else screen['validation']['loss'],
                full_loss=None if not done else done['validation']['loss'],
                reused_training=bool(job.get('reuse_from')), failed=bool(failure)))
            if candidate != chosen or not test:
                continue
            tests += 1
            new_tests += not test.get('reused_previous_test', False)
            value = test['metric']
            graph = job['graphlora_reference']['test']['metric']
            old = job['previous_gpf']['test']['metric']
            control = job['reference']['test']['metric']
            gain = graph - value if job['config']['regression'] else value - graph
            values.append(value); graphs.append(graph); olds.append(old); controls.append(control); gains.append(gain)
            pairs.append(dict(cell=cell, candidate=candidate, seed=seed, graphlora=graph,
                previous_gpf=old, refined_gpf=value, gain_over_graphlora=gain,
                reused_previous_test=test.get('reused_previous_test', False)))
        refs = [references[seed] for seed in study['seeds']]
        rows.append(dict(cell=cell, new_screened=cell_screened, selected_candidate=chosen,
            selection=selection or None, validation_fits=completed, tests=len(values),
            test=stats(values), graphlora=stats([j['graphlora_reference']['test']['metric'] for j in refs]),
            previous_gpf=stats([j['previous_gpf']['test']['metric'] for j in refs]),
            paired_gain_over_graphlora=stats(gains), paired_no_prompt=stats(controls)))
    expected_screens = len(study['cells']) * (len(study['candidates']) - 1)
    expected_fits = len(study['cells']) * study['promote'] * len(study['seeds'])
    expected_tests = 0 if study['smoke'] else study['final_tests']
    all_tests = len(list(root.glob('*/candidate_*/seed_*/test.json')))
    blocking = [f for f in failures if not f['numerical_rejection']]
    audit = read(root / 'completion_audit.json', {})
    status = dict(complete=bool(audit.get('passed')) and (study['smoke'] or tests == expected_tests)
                            and all_tests == tests and not blocking,
        new_screened=screened, new_screens_expected=expected_screens, reused_incumbent_fits=cached,
        validation_fits=validation_fits, validation_fits_expected=expected_fits,
        selected_cells=selected_cells, cells_expected=len(study['cells']),
        tests=tests, tests_expected=expected_tests, new_test_evaluations=new_tests,
        reused_test_results=tests-new_tests, seeds=study['seeds'], running=running,
        failures=failures, blocking_failures=blocking, smoke=study['smoke'])
    write_json(root / 'status.json', status)
    write_json(root / 'summary.json', rows)
    for name, records in (('validation_trials.csv', trials), ('resources.csv', resources), ('paired_differences.csv', pairs)):
        if records:
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader(); writer.writerows(records)
            _write_text(root / name, stream.getvalue())
    round_number = study.get('refinement_round', 2)
    screens_per_cell = len(study['candidates']) - 1
    fits_per_cell = study['promote'] * len(study['seeds'])
    screen_epochs = study.get('screen_epochs', 60)
    lines = [f'# TABLE1 GPF-LoRAP 第 {round_number} 轮调参', '',
        f"状态：{'完成' if status['complete'] else '存在阻塞失败' if blocking else '进行中'}。"
        f"新候选筛选 {screened}/{expected_screens}；入围配置三种子训练 {validation_fits}/{expected_fits}；"
        f"配置选择 {selected_cells}/{len(study['cells'])}；最终结果 {tests}/{expected_tests}。", '',
        f'每个组合新增 {screens_per_cell} 个固定候选，围绕学习率、调度器、权重衰减、提示强度和容量搜索。'
        f'图任务完整训练集筛选 {screen_epochs} 轮，节点任务直接跑满轮数；'
        f"各组合前 {study['promoted_new']} 个新候选与上一轮配置一起，"
        '完成三个种子的完整训练，按三个最佳验证损失的平均值锁定配置。平局按候选编号，'
        '检查点平局保留较早轮次。所有最终选择锁定后才释放测试。', '',
        '上一轮配置 r00 始终进入最终比较，其已有训练与测试结果直接复用。'
        '如果 r00 的平均验证损失更好，就保留 r00，不重复测试。新配置每个模型只测试一次。'
        '复用训练的时间单列，不计为新增 GPU 耗时。冒烟使用少量训练/验证图且不访问测试集。', '',
        '架构、宽度、位宽范围、存储预算和损失系数保持一致，仍为骨干和量化参数联合训练。'
        '这是已知历史测试结果之后的探索性继续调参，与固定 GraphLoRA-adapted 的调参预算不同。'
        '测试成绩不决定本轮晋级或配置选择，不按测试表现取新旧最大值。', '',
        '| 组合 | 新筛选 | 入围完整训练 | 选中配置 | 上一轮 GPF-LoRAP | GraphLoRA | 本轮 GPF-LoRAP |',
        '|---|---:|---:|---|---:|---:|---:|']
    if study.get('comparison_note'):
        lines[2:2] = [study['comparison_note'], '']
    tex = [f'% Round {round_number}: mean three-seed validation selection; tuned GPF-LoRAP vs fixed GraphLoRA-adapted.',
           '% Only fully evaluated seed triples; incumbent tests reused when selected.',
           f'Config & GraphLoRA-adapted & GPF-LoRAP (round {round_number})' + r' \\']
    for row in rows:
        digits = 4 if '_ZINC_' in row['cell'] else 2
        lines.append(f"| {row['cell']} | {row['new_screened']}/{screens_per_cell} | {row['validation_fits']}/{fits_per_cell} | "
            f"{row['selected_candidate'] or '待选'} | {formatted(row['previous_gpf'],digits)} | "
            f"{formatted(row['graphlora'],digits)} | {formatted(row['test'],digits)} |")
        if row['tests'] == len(study['seeds']) and not study['smoke']:
            vals = [formatted(row[key],digits).replace(' ± ',r' \pm ') for key in ('graphlora','test')]
            tex.append(row['cell'].replace('_',r'\_')+' & '+' & '.join('$'+v+'$' for v in vals)+r' \\')
    lines += ['', '分类报告准确率百分数，ZINC 报告 MAE；均值 ± 样本标准差。'
              '不足三个种子的测试数字只是临时结果。全部原始配置、代码快照、划分、验证记录、'
              '配对指标、参数量及实际量化位宽均保留。']
    if running:
        lines += ['', '正在运行：']
        for item in running:
            lines.append(f"- {item['cell']} / {item['candidate']} / seed {item['seed']}："
                f"{item.get('epoch',0)}/{item.get('cap','?')}，{item['device']}。")
    if failures:
        lines += ['', '失败记录：'] + [f"- {f['directory']}/failure.json" for f in failures]
    _write_text(root / 'REPORT.md', '\n'.join(lines)+'\n')
    _write_text(root / 'table1_refined_rows.tex', '\n'.join(tex)+'\n')
    return status
