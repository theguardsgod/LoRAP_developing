"""Saved-result reporting; no model execution or result-dependent selection."""
import csv
import io
import json
from pathlib import Path

from ..table1_frozen.common import file_lock, write_json
from ..table1_graphlora.report import formatted, stats


def read(path):
    return json.loads(path.read_text())


def generate(root):
    root = Path(root)
    with file_lock(root / '.report.lock'):
        return _generate(root)


def _generate(root):
    study = read(root / 'study.json')
    rows, resources, pairs, running, failures = [], [], [], [], []
    fits = tests = 0
    for cell in study['cells']:
        values, controls, graphs, gains, validation = [], [], [], [], []
        cell_fits = 0
        for job in (j for j in study['jobs'] if j['cell'] == cell):
            path = root / job['directory']
            try:
                worker = read(path / 'worker.json')
            except FileNotFoundError:
                worker = None
            if worker is not None:
                if (path / 'progress.json').exists():
                    worker.update(read(path / 'progress.json'))
                running.append(dict(cell=cell, seed=job['seed'], **worker))
            if (path / 'failure.json').exists():
                failures.append(job['directory'])
            if not (path / 'done.json').exists():
                continue
            done = read(path / 'done.json')
            fits += 1
            cell_fits += 1
            validation.append(done['validation']['metric'])
            resources.append(dict(cell=cell, seed=job['seed'],
                trainable_parameters=done['trainable_parameters'], total_parameters=done['total_parameters'],
                prompt_parameters=done['prompt_parameters'], prompt_KiB=done['adapter_fp32_bytes'] / 1024,
                training_seconds=done['training_seconds'], peak_cuda_MiB=done['peak_cuda_memory_bytes'] / 1024**2,
                device=done['device']))
            if (path / 'test.json').exists():
                test = read(path / 'test.json')['metric']
                base = job['reference']['test']['metric']
                graph = job.get('graphlora_reference', {}).get('test', {}).get('metric')
                gain = None if graph is None else (graph - test if job['config']['regression'] else test - graph)
                tests += 1
                values.append(test)
                controls.append(base)
                if graph is not None:
                    graphs.append(graph)
                    gains.append(gain)
                pairs.append(dict(cell=cell, seed=job['seed'], no_prompt=base, graphlora=graph,
                                  gpf_lorap=test, gain_over_graphlora=gain))
        rows.append(dict(cell=cell, fits=cell_fits, tests=len(values), validation=stats(validation),
                         test=stats(values), paired_no_prompt=stats(controls), paired_graphlora=stats(graphs),
                         paired_gain_over_graphlora=stats(gains),
                         historical=study.get('historical_table', {}).get(cell)))
    status = dict(complete=(root / 'completion_audit.json').exists() and read(root / 'completion_audit.json')['passed'],
                  fits=fits, fits_total=study['training_jobs'], tests=tests,
                  tests_expected=0 if study['smoke'] else study['training_jobs'],
                  seeds=study['seeds'], failures=failures, running=running)
    write_json(root / 'status.json', status)
    write_json(root / 'summary.json', rows)
    for filename, records in (('resources.csv', resources), ('paired_differences.csv', pairs)):
        if records:
            stream = io.StringIO()
            writer = csv.DictWriter(stream, fieldnames=list(records[0]))
            writer.writeheader()
            writer.writerows(records)
            (root / filename).write_text(stream.getvalue())
    lines = ['# TABLE1 GPF-LoRAP 定向重跑', '',
             f"训练 {fits}/{study['training_jobs']}，测试 {tests}/{status['tests_expected']}；"
             f"{'完成' if status['complete'] else '存在失败，请检查' if failures else '进行中'}。", '',
             f"种子：{study['seeds']}。固定一个配置：20 个提示，LoRAP rank=2；"
             '相同宽度共享打分网络，每个聚合层独立低秩因子；GPF-plus 加在输入投影之前，'
             'ZINC 加在原子嵌入之后。提示学习率等于骨干主学习率，权重衰减为零。', '',
             '沿用已完成 GraphLoRA-adapted 的骨干、量化器、初始化、完整划分、训练轮数、'
             '优化器基础分组和调度器。全部参数按既有规则联合训练；'
             'GPF-LoRAP 使用任务损失和 A2Q 存储惩罚，不附加 GraphLoRA 的方法专属损失。', '',
             '**实验范围按用户提供表格的历史结果选定。** 这是选定子集上的受控重跑；'
             '原表逐项完整运行配置未提供，因此不称为历史配方的精确复现。'
             '保留全部三种子结果和旧表记录，不按测试成绩在新旧结果中择优。', '',
             '| 组合 | 已训练 | 原表 GPF-LoRAP | 原表 GraphLoRA | 新 GPF-LoRAP 测试 | 配对 GraphLoRA 增益 |',
             '|---|---:|---:|---:|---:|---:|']
    tex = ['% Matched protocol, three seeds; accuracy (%) higher / ZINC MAE lower.',
           '% Historical values are comparison context, not paired controls.',
           r'Config & No prompt & GraphLoRA-adapted & GPF-LoRAP (rerun) \\']
    for row in rows:
        digits = 4 if '_ZINC_' in row['cell'] else 2
        old = row['historical']
        historical = lambda key: '未知' if old is None else f"{old[key]['mean']:.{digits}f} ± {old[key]['sd']:.{digits}f}"
        lines.append(f"| {row['cell']} | {row['fits']}/{len(study['seeds'])} | {historical('gpf_lorap')} | "
                     f"{historical('graphlora')} | {formatted(row['test'], digits)} | "
                     f"{formatted(row['paired_gain_over_graphlora'], digits)} |")
        # Emit only fully evaluated triples; partial results are never table-ready.
        if row['test'] is not None and row['test']['n'] == len(study['seeds']):
            vals = [formatted(row[k], digits).replace(' ± ', r' \pm ')
                    for k in ('paired_no_prompt', 'paired_graphlora', 'test')]
            tex.append(row['cell'].replace('_', r'\_') + ' & ' + ' & '.join('$' + v + '$' for v in vals) + r' \\')
    lines += ['', '分类以准确率百分数报告，ZINC 以 MAE 报告；均值 ± 样本标准差。'
              '增益为正表示新 GPF-LoRAP 更好；分类为百分点，ZINC 为 MAE 下降量。', '',
              '所有正式训练完成后，按最低验证任务损失选出的检查点各测试一次。'
              '冒烟不访问测试集。逐轮日志、完整配置、初始化指纹、代码快照、实际位宽、'
              '参数量、耗时及峰值显存分别保存。表中原有括号提升量需要用同协议无提示对照重新计算。']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    (root / 'table1_matched_rerun.tex').write_text('\n'.join(tex) + '\n')
    return status
