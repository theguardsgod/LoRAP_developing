"""Read saved records only; never run models or select hyperparameters."""
import csv
import io
import json
from pathlib import Path
import statistics

from ..table1_frozen.common import write_json
from ..table1_frozen.registry import DATASETS, ARCHITECTURES, FRAMEWORKS


def stats(values):
    return dict(mean=statistics.mean(values), sd=statistics.stdev(values) if len(values) > 1 else None, n=len(values)) if values else None


def formatted(summary, digits=2):
    if summary is None:
        return '待测'
    sd = f" ± {summary['sd']:.{digits}f}" if summary['sd'] is not None else ' (n=1)'
    return f"{summary['mean']:.{digits}f}" + sd


def generate(root):
    root = Path(root)
    study = json.loads((root / 'study.json').read_text())
    completed = tested = 0
    failures, rows, resource_rows, running = [], [], [], []
    for name in study['cells']:
        baseline, tests, validations, gains, params, kib = [], [], [], [], [], []
        for job in [j for j in study['jobs'] if j['cell'] == name]:
            path = root / job['directory']
            if (path / 'worker.json').exists():
                worker = json.loads((path / 'worker.json').read_text())
                if (path / 'progress.json').exists():
                    worker.update(json.loads((path / 'progress.json').read_text()))
                running.append(dict(cell=name, seed=job['seed'], **worker))
            if (path / 'failure.json').exists():
                failures.append(job['directory'])
            if not (path / 'done.json').exists():
                continue
            done = json.loads((path / 'done.json').read_text())
            completed += 1
            validations.append(done['validation']['metric'])
            params.append(done['adapter_parameters'])
            kib.append(done['adapter_fp32_bytes'] / 1024)
            resource_rows.append(dict(cell=name, seed=job['seed'], protocol=study['protocol'],
                adapter_parameters=done['adapter_parameters'], adapter_KiB=kib[-1],
                adapter_buffer_bytes=done['adapter_buffer_bytes'], total_parameters=done['total_parameters'],
                trainable_parameters=done['trainable_parameters'], training_seconds=done['training_seconds'],
                peak_cuda_MiB=done['peak_cuda_memory_bytes'] / 1024**2, device=done['device']))
            if (path / 'test.json').exists():
                test = json.loads((path / 'test.json').read_text())
                tested += 1
                tests.append(test['metric'])
                base = job['reference']['test']['metric']
                baseline.append(base)
                gains.append((base - test['metric']) if job['config']['regression'] else (test['metric'] - base))
        rows.append(dict(cell=name, fits=len(validations), tests=len(tests), validation=stats(validations),
                         test=stats(tests), paired_control=stats(baseline), paired_gain=stats(gains),
                         adapter_parameters=stats(params), adapter_KiB=stats(kib)))
    audit = root / 'completion_audit.json'
    status = dict(complete=audit.exists() and json.loads(audit.read_text())['passed'],
                  fits=completed, fits_total=study['training_jobs'], tests=tested,
                  tests_expected=0 if study['smoke'] else study['training_jobs'],
                  seeds=study['seeds'], protocol=study['protocol'], failures=failures, running=running)
    write_json(root / 'status.json', status)
    write_json(root / 'summary.json', rows)
    if resource_rows:
        stream = io.StringIO()
        writer = csv.DictWriter(stream, fieldnames=list(resource_rows[0]))
        writer.writeheader()
        writer.writerows(resource_rows)
        (root / 'resources.csv').write_text(stream.getvalue())
    lines = ['# TABLE1 GraphLoRA-adapted（三种子）', '',
             f"训练 {completed}/{study['training_jobs']}；测试 {tested}/{status['tests_expected']}；"
             f"状态：{'完成' if status['complete'] else '需检查失败' if failures else '进行中'}。", '',
             f"种子 {study['seeds']}；训练方式 {study['protocol']}；rank=2；"
             '复用已完成的同种子无提示对照，不重训、不重测对照。', '',
             '完整范围：5 数据集 × 3 骨干 × QAT/DQ/A2Q，共 45 cell、135 次新增训练。'
             '固定一个超参数配置，无新增搜索；旧 Table V 的 10 种子结果保留。', '',
             '**这是同域 GraphLoRA 移植**。GIN 使用两层低秩 MLP，GCN 使用低秩消息映射，'
             'GAT 使用独立低秩注意力分支。无源域、无 SMMD。'
             '节点对比监督仅使用训练节点标签；ZINC 无类别正样本，结构约束使用归一化节点表示。', '',
             'joint 为从相同初始化联合训练；frozen 为从相同已训练检查点冻结骨干。'
             '必须结合此标签比较，不能把两种训练协议混称为冻结 PEFT。', '',
             '| 组合 | 完成训练 | 验证指标 | 最终测试 | 配对无适配测试 | 平均增益 | 参数 KiB |',
             '|---|---:|---:|---:|---:|---:|---:|']
    for row in rows:
        digits = 4 if '_ZINC_' in row['cell'] else 2
        lines.append(f"| {row['cell']} | {row['fits']}/{len(study['seeds'])} | "
                     f"{formatted(row['validation'], digits)} | {formatted(row['test'], digits)} | "
                     f"{formatted(row['paired_control'], digits)} | {formatted(row['paired_gain'], digits)} | "
                     f"{formatted(row['adapter_KiB'])} |")
    lines += ['', '分类指标为准确率百分数，ZINC 为 MAE；增益统一为正值表示改善。'
              '均值 ± 样本标准差。3 种子为探索性重复，不能视作原表 10/100 次实验。', '',
              '所有训练完成后才释放测试；按最低验证任务损失选模，平局保留最早轮。'
              'smoke 不访问任何真实测试集。额外参数均为 FP32，BN 缓冲区另列。'
              '各模型实际量化器位宽保存于 quantization.json；不冒称其为原表相同位宽定义。']
    (root / 'REPORT.md').write_text('\n'.join(lines) + '\n')
    tex = ['% GraphLoRA-adapted; ' + study['protocol'] + '; seeds=' + ','.join(map(str, study['seeds'])),
           '% Columns: framework, backbone, method, Cora, CiteSeer, MNIST, CIFAR10, ZINC.',
           '% Classification: accuracy (%); ZINC: MAE. Mean +/- sample standard deviation.']
    lookup = {row['cell']: row for row in rows}
    for framework in FRAMEWORKS:
        for architecture in ARCHITECTURES:
            values = []
            for dataset in DATASETS:
                row = lookup.get(f'{framework}_{dataset}_{architecture}')
                value = None if row is None else row['test']
                if value is None or value['n'] != len(study['seeds']):
                    values.append('TBD')
                else:
                    digits = 4 if dataset == 'ZINC' else 2
                    values.append('$' + formatted(value, digits).replace(' ± ', r' \pm ') + '$')
            tex.append(' & '.join([framework, architecture, 'GraphLoRA-adapted', *values]) + r' \\')
    (root / 'table1_graphlora_rows.tex').write_text('\n'.join(tex) + '\n')
    return status
