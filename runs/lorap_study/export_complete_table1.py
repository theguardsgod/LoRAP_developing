"""Assemble the supplied Table I and latest completed reruns, with provenance.

No model inference. Later validation-selected rounds replace earlier rounds;
the exporter does not choose the largest test score across attempts.
"""
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


REPO = Path(__file__).resolve().parents[2]
BASE = REPO / 'result/lorap_study'
OUTPUT = BASE / 'table1_complete_20261019_v1'
ORIGINAL = BASE / 'table1_gpf_rerun_3seeds_20261001_v1/input_table.tex'
GRAPH = BASE / 'table1_graphlora_3seeds_20260925_v1'
ROUNDS = [
    ('rerun', 'table1_gpf_rerun_3seeds_20261001_v1'),
    ('tuned', 'table1_gpf_search_3seeds_20261007_v1'),
    ('tuned', 'table1_gpf_refine_3seeds_20261011_v1'),
    ('tuned', 'table1_gpf_refine3_3seeds_20261013_v1'),
    ('expanded', 'table1_gpf_refine4_3seeds_20261013_v1'),
]
DATASETS = ['Cora', 'CiteSeer', 'MNIST', 'CIFAR10', 'ZINC']
METHODS = ['None', 'GPF-plus', 'GraphLoRA-adapted', 'GPF-LoRAP']
NUMBER = re.compile(r'([0-9]+(?:\.[0-9]+)?)\\pm([0-9]+(?:\.[0-9]+)?)')


def read(path):
    return json.loads(path.read_text())


def historical_rows(text):
    chunks = re.split(r'% (QAT|DQ|A2Q) \+ (GIN|GCN|GAT)', text)
    rows = []
    for fragment in chunks[0].split(r'\\'):
        values = NUMBER.findall(fragment)
        if len(values) == 5:
            arch = fragment.split('&')[1].strip()
            rows.append(('FP32', arch, 'None', values))
    for i in range(1, len(chunks), 3):
        framework, architecture, body = chunks[i:i + 3]
        for fragment in body.split(r'\\'):
            values = NUMBER.findall(fragment)
            if not values:
                continue
            assert len(values) == 5
            label = fragment.split('&')[2]
            method = next(name for name in ('GPF-LoRAP', 'GPF-plus', 'GraphLoRA', 'None') if name in label)
            if method == 'GraphLoRA':
                method = 'GraphLoRA-adapted'
            rows.append((framework, architecture, method, values))
    assert len(rows) == 39
    assert len({(f, a, m) for f, a, m, _ in rows}) == 39
    return rows


def completed(root):
    study = read(root / 'study.json')
    assert read(root / 'completion_audit.json')['passed']
    assert not study['smoke'] and study['seeds'] == [41, 42, 43]
    return study, {r['cell']: r for r in read(root / 'summary.json')}


def verified_metrics(root, study, summary, cell):
    selected = summary[cell].get('selected_candidate')
    jobs = [j for j in study['jobs'] if j['cell'] == cell and
            (selected is None or j.get('candidate') == selected)]
    assert sorted(j['seed'] for j in jobs) == [41, 42, 43]
    if selected is not None:
        assert read(root / 'SELECTION_LOCK.json')['cells'][cell]['candidate'] == selected
    values, artifacts = [], []
    for job in jobs:
        path = root / job['directory']
        done, test = read(path / 'done.json'), read(path / 'test.json')
        assert test['completed_test_evaluations'] == 1
        assert test['checkpoint_file_sha256'] == done['checkpoint_file_sha256']
        values.append(test['metric'])
        artifacts.append(dict(seed=job['seed'], metric=test['metric'], epochs=job['epochs'],
                              test_file=str(path / 'test.json'), config_file=str(path / 'config.json')))
    mean, sd = statistics.mean(values), statistics.stdev(values)
    expected = summary[cell]['test']
    assert expected['n'] == 3
    assert math.isclose(mean, expected['mean'], abs_tol=1e-12)
    assert math.isclose(sd, expected['sd'], abs_tol=1e-12)
    digits = 4 if '_ZINC_' in cell else 2
    return dict(mean=mean, sd=sd, mean_text=f'{mean:.{digits}f}', sd_text=f'{sd:.{digits}f}',
                seeds=[41, 42, 43], candidate=selected, source=str(root), artifacts=artifacts)


def export():
    original_text = ORIGINAL.read_text()
    rows = historical_rows(original_text)
    graph_study, graph_summary = completed(GRAPH)
    graph = {cell: verified_metrics(GRAPH, graph_study, graph_summary, cell) for cell in graph_study['cells']}
    assert len(graph) == 45
    latest = {}
    for kind, name in ROUNDS:
        root = BASE / name
        study, summary = completed(root)
        for cell in study['cells']:
            latest[cell] = dict(verified_metrics(root, study, summary, cell), kind=kind)
    assert len(latest) == 10
    records = []
    for framework, arch, method, values in rows:
        for dataset, (mean, sd) in zip(DATASETS, values):
            cell = f'{framework}_{dataset}_{arch}'
            metric = dict(mean=float(mean), sd=float(sd), mean_text=mean, sd_text=sd,
                          source=str(ORIGINAL), kind='historical', seeds=None, artifacts=[])
            if method == 'GraphLoRA-adapted':
                metric = dict(graph[cell], kind='graphlora_fixed')
                assert float(mean) == float(metric['mean_text']) and float(sd) == float(metric['sd_text'])
            elif method == 'GPF-LoRAP' and cell in latest:
                metric = latest[cell]
            records.append(dict(framework=framework, architecture=arch, method=method,
                                dataset=dataset, cell=cell, **metric))
    assert len(records) == 195
    assert len([r for r in records if r['kind'] in ('rerun', 'tuned', 'expanded')]) == 10
    for record in records:
        if record['framework'] == 'FP32':
            record['rank'] = None
        else:
            peers = [r['mean'] for r in records if r['cell'] == record['cell']]
            means = sorted(set(peers), reverse=record['dataset'] != 'ZINC')
            record['rank'] = means.index(record['mean']) + 1

    def tex_value(record):
        value = record['mean_text'] + r'\pm' + record['sd_text']
        if record['rank'] == 1:
            value = r'\mathbf{' + value + '}'
        elif record['rank'] == 2:
            value = r'\mathit{' + value + '}'
        mark = {'rerun': r'^{\dagger}', 'tuned': r'^{\ddagger}', 'expanded': r'^{\ddagger *}'}.get(record['kind'], '')
        return '$' + value + mark + '$'

    def md_value(record):
        value = record['mean_text'] + ' ± ' + record['sd_text']
        if record['rank'] == 1:
            value = '**' + value + '**'
        elif record['rank'] == 2:
            value = '*' + value + '*'
        return value + {'rerun': '†', 'tuned': '‡', 'expanded': '‡⋆'}.get(record['kind'], '')

    lines = [r'% Required packages: \usepackage{booktabs,multirow,graphicx}',
        r'\begin{table*}[t]', r'\centering', r'\footnotesize', r'\setlength{\tabcolsep}{4pt}',
        r'\caption{Results under QAT, Degree-Quant (DQ), and $A^2Q$, with FP32 references.',
        r'Bold and italic indicate the best and second-best reported means within each framework--architecture--dataset block.',
        r'This table combines historical results with new reruns and tuning results; see the source and protocol notes below.}',
        r'\label{tab:results}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{lllccccc}', r'\toprule',
        r'Quant. & Model & Adaptation & \multicolumn{2}{c}{Node classification (Acc. \%)} & \multicolumn{2}{c}{Graph classification (Acc. \%)} & Graph regression \\',
        r'Frame. & Arch. & Method & Cora $\uparrow$ & CiteSeer $\uparrow$ & MNIST $\uparrow$ & CIFAR-10 $\uparrow$ & ZINC MAE $\downarrow$ \\',
        r'\midrule']
    markdown = ['# 完整 Table I（截至 2026-10-19）', '',
        '保留用户原表的 FP32、None、GPF-plus 及未重跑 GPF-LoRAP 数据；GraphLoRA-adapted 来自已完成实验，10 个 GPF-LoRAP 单元格更新为各组合最后完成轮次的验证集选定配置。', '',
        '| 量化 | 模型 | 方法 | Cora ↑ | CiteSeer ↑ | MNIST ↑ | CIFAR-10 ↑ | ZINC MAE ↓ |',
        '|---|---|---|---:|---:|---:|---:|---:|']
    for index, (framework, arch, method, _) in enumerate(rows):
        subset = [r for r in records if (r['framework'], r['architecture'], r['method']) == (framework, arch, method)]
        assert [r['dataset'] for r in subset] == DATASETS
        if framework == 'FP32':
            prefix = (r'\multirow{3}{*}{FP32}' if index == 0 else '') + ' & ' + arch + ' & None'
        else:
            if method == 'None':
                if index == 3 or rows[index - 1][1] != arch:
                    lines.append(r'\midrule')
                lines += [r'\midrule', f'% {framework} + {arch}']
                f = r'$A^2Q$' if framework == 'A2Q' else framework
                prefix = r'\multirow{4}{*}{' + f + r'} & \multirow{4}{*}{' + arch + '} & ' + method
            else:
                prefix = '& & ' + method
        lines.append(prefix + ' & ' + ' & '.join(tex_value(r) for r in subset) + r' \\')
        markdown.append('| ' + ' | '.join([framework, arch, method] + [md_value(r) for r in subset]) + ' |')
    lines += [r'\bottomrule', r'\end{tabular}%', '}', r'\par\vspace{3pt}',
        r'\begin{minipage}{\textwidth}', r'\scriptsize',
        r'\textit{Notes.} FP32, None, GPF-plus, and unmarked GPF-LoRAP entries are retained from the original table.',
        r'GraphLoRA-adapted and all marked GPF-LoRAP entries report means $\pm$ sample standard deviations over seeds 41, 42, and 43.',
        r'$\dagger$: fixed-configuration rerun. $\ddagger$: validation-selected result after additional tuning.',
        r'$*$: QAT--GCN--CIFAR-10 uses 300 training epochs and altered regularization; its fixed GraphLoRA-adapted reference uses 200 epochs.',
        r'The new runs use the current implementation and are not exact reproductions of the historical training recipes.',
        r'Tuning budgets differ, so numerical ranks across sources do not establish a matched-protocol comparison or statistical significance.',
        r'Parenthesized gains are omitted because the historical None entries are not the paired controls of the new runs.',
        r'\end{minipage}', r'\end{table*}']
    notes = ('† 固定配置重跑；‡ 验证集选定的追加调参结果；⋆ QAT–GCN–CIFAR10 使用 300 轮及调整后的正则化，固定 GraphLoRA 对照为 200 轮。'
             'GraphLoRA-adapted 和所有带标记结果均为种子 41/42/43 的均值 ± 样本标准差。'
             '历史数据与新实验的实现/训练设置不完全相同，搜索预算不同，表内数值排名不代表同协议比较或统计显著性。'
             '原表括号增益已移除，因为历史 None 不是新实验的配对对照。')
    markdown += ['', notes, '', '10 个 GPF-LoRAP 更新单元格及来源：', '']
    for cell, metric in latest.items():
        markdown.append(f"- {cell}: {metric['mean_text']} ± {metric['sd_text']}，来源 {Path(metric['source']).name}，配置 {metric.get('candidate') or '固定配置'}。")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / 'original_table.tex').write_text(original_text)
    (OUTPUT / 'table1_complete.tex').write_text('\n'.join(lines) + '\n')
    (OUTPUT / 'table1_complete.md').write_text('\n'.join(markdown) + '\n')
    (OUTPUT / 'provenance.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    with (OUTPUT / 'table1_complete.csv').open('w', newline='') as stream:
        fields = ['framework', 'architecture', 'method', 'dataset', 'mean', 'sd', 'mean_text', 'sd_text', 'kind', 'rank', 'source', 'candidate']
        writer = csv.DictWriter(stream, fields, extrasaction='ignore')
        writer.writeheader(); writer.writerows(records)
    wrapper = [r'\documentclass[twocolumn]{article}', r'\usepackage[margin=1.4cm]{geometry}',
               r'\usepackage{booktabs,multirow,graphicx}', r'\begin{document}',
               r'\input{table1_complete.tex}', r'\end{document}']
    (OUTPUT / 'table1_standalone.tex').write_text('\n'.join(wrapper) + '\n')
    audit = dict(passed=True, data_rows=len(rows), metric_cells=len(records), graphlora_cells=45,
        updated_gpf_cells=len(latest), verified_final_test_records=165,
        update_policy='latest completed validation-selected round per targeted cell; not maximum test score',
        original_table_sha256=hashlib.sha256(original_text.encode()).hexdigest(),
        latex_compiled=False, latex_note='No TeX engine is installed; numeric and structural checks passed.',
        updated_gpf={k:dict(mean=v['mean'],sd=v['sd'],source=v['source'],candidate=v.get('candidate')) for k,v in latest.items()})
    (OUTPUT / 'export_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(output=str(OUTPUT), rows=len(rows), metrics=len(records), updated_gpf_cells=len(latest)), indent=2))


if __name__ == '__main__':
    export()
