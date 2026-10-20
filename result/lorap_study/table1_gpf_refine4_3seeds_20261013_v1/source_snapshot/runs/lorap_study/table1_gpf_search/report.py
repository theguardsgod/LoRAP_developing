"""Report saved search records without running models or selecting candidates."""
import csv
import io
import json
import os
from pathlib import Path

from ..table1_frozen.common import file_lock, write_json
from ..table1_graphlora.report import formatted, stats


def read(path, default=None):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return default


def _write_text(path, value):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    temporary.write_text(value, encoding='utf-8')
    os.replace(temporary, path)


def _write_csv(path, records, fields):
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=fields)
    writer.writeheader()
    writer.writerows(records)
    _write_text(path, stream.getvalue())


def _metric(job, key):
    return (job.get(key) or {}).get('test', {}).get('metric')


def _mean_metrics(jobs, key):
    return stats([value for job in jobs if (value := _metric(job, key)) is not None])


def _validation(record, key):
    return None if record is None else record.get('validation', {}).get(key)


def generate(root):
    root = Path(root)
    if read(root / 'study.json').get('selection_protocol') == 'mean_three':
        from .refine_report import generate as generate_refine
        return generate_refine(root)
    with file_lock(root / '.report.lock'):
        return _generate(root)


def _generate(root):
    study = read(root / 'study.json')
    seeds = study['seeds']
    search_seed = study.get('search_seed', seeds[0])
    smoke = study.get('smoke', False)
    rows, resources, pairs, running, failures, trials, selections = [], [], [], [], [], [], []
    numerical_rejections, blocking_failures = [], []
    screened = full_fits = promoted_full_fits = confirmation_fits = tests = all_tests = 0
    completed_screen_attempts = completed_promoted_attempts = 0
    screen_expected = promoted_expected = 0
    for cell in study['cells']:
        jobs = [job for job in study['jobs'] if job['cell'] == cell]
        candidates = list(dict.fromkeys(job['candidate'] for job in jobs))
        promotion = read(root / cell / 'promotion.json')
        promoted = [] if promotion is None else promotion['candidates']
        selection = read(root / cell / 'selection.json')
        selected = None if selection is None else selection['candidate']
        if selection is not None:
            selections.append(dict(selection, cell=cell))
        screen_expected += sum(job['seed'] == search_seed for job in jobs)
        promoted_expected += len(promoted) if promotion is not None else min(
            len(candidates), study.get('promote', study.get('promotion_count', study.get('promote_count', 2))))
        cell_screened = cell_full = cell_promoted_full = cell_confirmed = 0
        cell_screen_attempts = cell_promoted_attempts = 0
        cell_rejections = []
        values, graphs, originals, controls, graph_gains, original_gains, validation = [], [], [], [], [], [], []
        reference_by_seed = {}
        for job in jobs:
            reference_by_seed.setdefault(job['seed'], job)
            candidate, seed = job['candidate'], job['seed']
            path = root / job['directory']
            screen = read(path / 'screen.json')
            done = read(path / 'done.json')
            test = read(path / 'test.json')
            worker = read(path / 'worker.json')
            if worker is not None:
                worker.update(read(path / 'progress.json', {}))
                running.append(dict(worker, cell=cell, candidate=candidate, seed=seed,
                                    directory=job['directory']))
            failure = read(path / 'failure.json')
            failed = failure is not None
            numerical_rejection = bool(failure and failure.get('numerical_rejection') and seed == search_seed)
            if failed:
                failures.append(job['directory'])
                if numerical_rejection:
                    numerical_rejections.append(job['directory'])
                    cell_rejections.append(job['directory'])
                else:
                    blocking_failures.append(job['directory'])
            if seed == search_seed and (screen is not None or numerical_rejection):
                completed_screen_attempts += 1
                cell_screen_attempts += 1
            if seed == search_seed and candidate in promoted and (done is not None or numerical_rejection):
                completed_promoted_attempts += 1
                cell_promoted_attempts += 1
            if screen is not None and seed == search_seed:
                screened += 1
                cell_screened += 1
            if done is not None:
                if seed == search_seed:
                    full_fits += 1
                    cell_full += 1
                    if candidate in promoted:
                        promoted_full_fits += 1
                        cell_promoted_full += 1
                else:
                    confirmation_fits += 1
                    cell_confirmed += 1
            record = done if done is not None else screen if screen is not None else failure
            if record is not None:
                count = read(path / 'initialization.json', {}) | record.get('counts', {}) | record
                adapter_bytes = count.get('adapter_fp32_bytes')
                peak_bytes = count.get('peak_cuda_memory_bytes')
                resources.append(dict(
                    cell=cell, candidate=candidate, seed=seed,
                    stage=('full' if done is not None else 'numerical_rejection' if numerical_rejection
                           else 'failure' if failed else 'screen'), selected=candidate == selected,
                    trainable_parameters=count.get('trainable_parameters'),
                    total_parameters=count.get('total_parameters'),
                    prompt_parameters=count.get('prompt_parameters', count.get('adapter_parameters')),
                    prompt_KiB=None if adapter_bytes is None else adapter_bytes / 1024,
                    training_seconds=count.get('training_seconds'),
                    peak_cuda_MiB=None if peak_bytes is None else peak_bytes / 1024**2,
                    device=count.get('device'), directory=job['directory']))
            trials.append(dict(
                cell=cell, candidate=candidate, seed=seed,
                screen_epochs=job.get('screen_epochs'), full_epochs=job['epochs'],
                screened=screen is not None, screen_validation_loss=_validation(screen, 'loss'),
                screen_validation_metric=_validation(screen, 'metric'),
                promoted=candidate in promoted, full_fit_completed=done is not None,
                full_validation_loss=_validation(done, 'loss'), full_validation_metric=_validation(done, 'metric'),
                selected_epoch=None if done is None else done.get('selected_epoch'),
                selected=candidate == selected, tested=test is not None, failed=failed,
                numerical_rejection=numerical_rejection,
                hyperparameters=json.dumps(job.get('hyperparameters', {}), sort_keys=True, ensure_ascii=False),
                config=json.dumps(job['config'], sort_keys=True, ensure_ascii=False),
                directory=job['directory']))
            all_tests += test is not None
            if candidate != selected or seed not in seeds:
                continue
            if done is not None and (loss := _validation(done, 'loss')) is not None:
                validation.append(loss)
            if test is None:
                continue
            metric = test['metric']
            graph, original, control = (_metric(job, key) for key in
                                        ('graphlora_reference', 'previous_gpf', 'reference'))
            sign = -1 if job['config']['regression'] else 1
            graph_gain = None if graph is None else sign * (metric - graph)
            original_gain = None if original is None else sign * (metric - original)
            tests += 1
            values.append(metric)
            if graph is not None:
                graphs.append(graph)
                graph_gains.append(graph_gain)
            if original is not None:
                originals.append(original)
                original_gains.append(original_gain)
            if control is not None:
                controls.append(control)
            pairs.append(dict(cell=cell, candidate=candidate, seed=seed, no_prompt=control,
                              graphlora=graph, previous_gpf_lorap=original, tuned_gpf_lorap=metric,
                              gain_over_graphlora=graph_gain, gain_over_previous_gpf=original_gain))
        references = [reference_by_seed[seed] for seed in seeds if seed in reference_by_seed]
        rows.append(dict(
            cell=cell, selected_candidate=selected, selection=selection, promotion=promotion,
            screened=cell_screened, screens_expected=sum(job['seed'] == search_seed for job in jobs),
            completed_screen_attempts=cell_screen_attempts,
            search_full_fits=cell_full, promoted_full_fits=cell_promoted_full,
            completed_promoted_attempts=cell_promoted_attempts, numerical_rejections=cell_rejections,
            confirmation_fits=cell_confirmed, tests=len(values),
            table_ready=len(values) == len(seeds), validation_loss=stats(validation), test=stats(values),
            previous_gpf=_mean_metrics(references, 'previous_gpf'),
            graphlora=_mean_metrics(references, 'graphlora_reference'),
            no_prompt=_mean_metrics(references, 'reference'),
            paired_no_prompt=stats(controls), paired_graphlora=stats(graphs),
            paired_previous_gpf=stats(originals), paired_gain_over_graphlora=stats(graph_gains),
            paired_gain_over_previous_gpf=stats(original_gains)))
    expected_tests = 0 if smoke else len(study['cells']) * len(seeds)
    audit = read(root / 'completion_audit.json', {})
    status = dict(
        complete=(bool(audit.get('passed')) and (smoke or tests == expected_tests)
                  and all_tests == tests and not blocking_failures),
        screened=screened, screens_expected=screen_expected,
        completed_screen_attempts=completed_screen_attempts,
        search_full_fits=full_fits,
        promoted_full_fits=promoted_full_fits, promoted_full_fits_expected=promoted_expected,
        completed_promoted_attempts=completed_promoted_attempts,
        confirmation_fits=confirmation_fits,
        confirmation_fits_expected=len(study['cells']) * sum(seed != search_seed for seed in seeds),
        fits=full_fits + confirmation_fits, tests=tests, tests_expected=expected_tests,
        all_test_records=all_tests, unexpected_test_records=all_tests - tests,
        selected_cells=len(selections), cells_expected=len(study['cells']),
        seeds=seeds, search_seed=search_seed, smoke=smoke,
        failures=failures, blocking_failures=blocking_failures, numerical_rejections=numerical_rejections,
        running=running, selections=selections)
    write_json(root / 'status.json', status)
    write_json(root / 'summary.json', rows)
    _write_csv(root / 'resources.csv', resources, [
        'cell', 'candidate', 'seed', 'stage', 'selected', 'trainable_parameters', 'total_parameters',
        'prompt_parameters', 'prompt_KiB', 'training_seconds', 'peak_cuda_MiB', 'device', 'directory'])
    _write_csv(root / 'validation_trials.csv', trials, [
        'cell', 'candidate', 'seed', 'screen_epochs', 'full_epochs', 'screened',
        'screen_validation_loss', 'screen_validation_metric', 'promoted', 'full_fit_completed',
        'full_validation_loss', 'full_validation_metric', 'selected_epoch', 'selected', 'tested',
        'failed', 'numerical_rejection', 'hyperparameters', 'config', 'directory'])
    _write_csv(root / 'paired_differences.csv', pairs, [
        'cell', 'candidate', 'seed', 'no_prompt', 'graphlora', 'previous_gpf_lorap', 'tuned_gpf_lorap',
        'gain_over_graphlora', 'gain_over_previous_gpf'])
    state = ('完成' if status['complete'] else '存在阻塞失败，请检查' if blocking_failures
             else '存在非预期测试记录，请检查' if status['unexpected_test_records'] else '进行中')
    lines = [
        '# TABLE1 GPF-LoRAP 验证集调参', '',
        f"状态：{state}。筛选尝试 {completed_screen_attempts}/{screen_expected}（有效结果 {screened}）；"
        f"晋级尝试 {completed_promoted_attempts}/{promoted_expected}（完整训练 {promoted_full_fits}）；"
        f"其余种子确认 {confirmation_fits}/"
        f"{status['confirmation_fits_expected']}；最终测试 {tests}/{expected_tests}。", '',
        f'数值不稳定排除 {len(numerical_rejections)} 个搜索候选；其他失败 {len(blocking_failures)} 个。'
        '搜索种子发生非有限数值的候选按既定规则排除，其记录保留且计入已尝试数量；'
        '单个候选的数值排除不代表整个搜索失败。基础设施错误和确认种子失败仍需处理。', '',
        f"搜索种子为 {search_seed}，最终报告种子为 {seeds}。每个候选先在搜索种子上短程训练，"
        '仅按验证任务损失决定晋级；晋级候选完成规定轮数后，再仅按验证任务损失选定每个组合的配置。'
        '选定配置随后完成其余种子训练，保留全部三种子结果。检查点同样仅按验证任务损失选择。', '',
        f'搜索种子已有 {full_fits} 次完整训练。节点任务可将筛选轮数设为完整训练轮数，'
        '因此搜索完整训练次数可能多于晋级候选数量；二者分别统计，筛选结果仍按同一验证损失规则晋级。', '',
        '**这是调参后的 GPF-LoRAP 与固定配置 GraphLoRA-adapted 的比较。** '
        '实验组合来自此前落后的子集，因此不能将本次结果描述为全部组合上的无偏比较或相等调参预算比较。'
        '测试集不参与候选晋级、配置选择或检查点选择；不按测试结果在新旧结果中择优。'
        '提高性能是搜索目标，不保证超过 GraphLoRA。', '',
        '正式实验使用既有完整数据划分。骨干架构、宽度、层数及原始量化位宽目标和预算保持不变；'
        '本轮注册候选调整提示超参数和学习率，权重衰减保持既有设置。每个候选的实际配置和超参数'
        '以 study.json 中 job 为准，均保存在 validation_trials.csv；保留初始化指纹及代码快照。', '',
        '**以下测试结果不足三个种子时均为临时结果，不可填入论文最终表格。** '
        'LaTeX 文件仅包含选定配置已完成全部种子测试的组合。冒烟检查不访问测试集。', '',
        '| 组合 | 选择配置 | 筛选尝试 | 晋级完整训练 | 确认 | 测试 | 原 GPF-LoRAP | 固定 GraphLoRA | 调参 GPF-LoRAP | 相对 GraphLoRA 增益 |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    tex = [
        '% Validation-tuned GPF-LoRAP versus fixed GraphLoRA-adapted; unequal tuning budgets.',
        '% Include only the selected candidate with every registered final seed evaluated.',
        '% Accuracy (%) is higher-is-better; ZINC MAE is lower-is-better; sample standard deviation.',
        r'Config & No prompt & GraphLoRA-adapted & GPF-LoRAP (original) & GPF-LoRAP (tuned) \\']
    for row in rows:
        digits = 4 if '_ZINC_' in row['cell'] else 2
        result = formatted(row['test'], digits)
        if row['tests'] and not row['table_ready']:
            result += f"（临时 n={row['tests']}）"
        lines.append(
            f"| {row['cell']} | {row['selected_candidate'] or '待选'} | "
            f"{row['completed_screen_attempts']}/{row['screens_expected']} | {row['promoted_full_fits']} | "
            f"{row['confirmation_fits']} | {row['tests']}/{len(seeds)} | "
            f"{formatted(row['previous_gpf'], digits)} | {formatted(row['graphlora'], digits)} | "
            f"{result} | {formatted(row['paired_gain_over_graphlora'], digits)} |")
        if row['table_ready'] and not smoke:
            values = [formatted(row[key], digits).replace(' ± ', r' \pm ')
                      for key in ('paired_no_prompt', 'paired_graphlora', 'paired_previous_gpf', 'test')]
            tex.append(row['cell'].replace('_', r'\_') + ' & ' +
                       ' & '.join('$' + value + '$' for value in values) + r' \\')
    lines += ['', '分类为准确率百分数，ZINC 为 MAE；均值 ± 样本标准差。正增益表示新配置更好：'
              '分类取新准确率减 GraphLoRA 准确率，ZINC 取 GraphLoRA MAE 减新 MAE。'
              '逐种子配对差值见 paired_differences.csv，参数量、FP32 开销、耗时和峰值显存见 resources.csv。']
    if running:
        lines += ['', '## 正在运行', '']
        for item in running:
            epoch = f"，epoch {item['epoch']}/{item.get('epochs', '?')}" if 'epoch' in item else ''
            lines.append(f"- {item['cell']} / {item['candidate']} / seed {item['seed']}："
                         f"{item.get('device', '设备待记录')}{epoch}。")
    if numerical_rejections:
        lines += ['', '## 数值不稳定排除的搜索候选', ''] + [
            f'- {directory}/failure.json' for directory in numerical_rejections]
    if blocking_failures:
        lines += ['', '## 需处理的失败记录', ''] + [f'- {directory}/failure.json' for directory in blocking_failures]
    if status['unexpected_test_records']:
        lines += ['', f"注意：发现 {status['unexpected_test_records']} 条不属于当前最终选择的测试记录，"
                  '未纳入最终指标，请核对测试门禁。']
    _write_text(root / 'REPORT.md', '\n'.join(lines) + '\n')
    _write_text(root / 'table1_tuned_rows.tex', '\n'.join(tex) + '\n')
    return status
