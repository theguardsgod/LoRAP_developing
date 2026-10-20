"""Read-only paired GraphLoRA progress and validation/test/resource reporting."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from .run import write_json


def read(path):
    return json.loads(path.read_text())


def stats(values):
    return None if not values else dict(mean=statistics.mean(values),
        sample_std=statistics.stdev(values) if len(values)>1 else None, n=len(values))


def show(value, digits=2):
    if value is None:return '待测'
    if value['sample_std'] is None:return f"{value['mean']:.{digits}f} (n=1)"
    return f"{value['mean']:.{digits}f} ± {value['sample_std']:.{digits}f}"


def generate_report(root):
    root = Path(root); study = read(root/'study.json')
    results, progress, rows, pairs = [], [], [], []
    expected = sum(len(c['seeds'])*len(c['variants']) for c in study['configs'])
    for config in study['configs']:
        mode = config['quantization_mode']; local = {}
        for seed in config['seeds']:
            for variant in config['variants']:
                directory = root/mode/'MNIST'/variant/f'seed_{seed}'
                if (directory/'progress.json').exists():
                    progress.append(dict(mode=mode, variant=variant, seed=seed, **read(directory/'progress.json')))
                if (directory/'result.json').exists():
                    result = dict(read(directory/'result.json'), mode=mode)
                    results.append(result); local[(seed, variant)] = result
        for variant in config['variants']:
            chosen = [r for (s, v), r in local.items() if v==variant]
            tested = [r for r in chosen if r['test_evaluations']==1]
            rows.append(dict(mode=mode, variant=variant, fits=len(chosen), tests=len(tested), seeds=len(config['seeds']),
                validation_accuracy=stats([100*r['validation']['accuracy'] for r in chosen]),
                test_accuracy=stats([100*r['test']['accuracy'] for r in tested]) if len(tested)==len(config['seeds']) else None,
                adapter_parameters=stats([r['adapter_parameters'] for r in chosen]),
                adapter_KiB=stats([r['adapter_fp32_bytes']/1024 for r in chosen]),
                effective_bits=stats([r['mean_activation_effective_bits'] for r in chosen])))
        for seed in config['seeds']:
            if (seed, 'no_prompt') not in local or (seed, 'graphlora') not in local:continue
            base, adapted = local[(seed, 'no_prompt')], local[(seed, 'graphlora')]
            assert base['common_backbone_sha256']==adapted['common_backbone_sha256']
            assert base['config_sha256']==adapted['config_sha256']
            pairs.append(dict(mode=mode, seed=seed,
                no_prompt_validation_percent=100*base['validation']['accuracy'],
                graphlora_validation_percent=100*adapted['validation']['accuracy'],
                validation_difference_pp=100*(adapted['validation']['accuracy']-base['validation']['accuracy']),
                test_difference_pp=100*(adapted['test']['accuracy']-base['test']['accuracy'])
                if base['test_evaluations'] and adapted['test_evaluations'] else None))
    tests = sum(r['test_evaluations'] for r in results)
    active = read(root/'active.json') if (root/'active.json').exists() else dict(phase='preparing')
    audit = read(root/'completion_audit.json') if (root/'completion_audit.json').exists() else {}
    complete = bool(audit.get('passed')) and len(results)==expected and (study['profile']!='full' or tests==expected)
    write_json(root/'status.json', dict(complete=complete, profile=study['profile'], active=active,
        fits_completed=len(results), fits_total=expected, tests_completed=tests,
        tests_expected=expected if study['profile']=='full' else 0, rows=rows, progress=progress))
    write_json(root/'comparison.json', rows)
    resource_fields = ['mode', 'variant', 'seed', 'epochs', 'best_epoch', 'total_parameters', 'trainable_parameters',
        'adapter_parameters', 'adapter_fp32_bytes', 'adapter_buffer_bytes', 'mean_activation_effective_bits',
        'training_seconds', 'peak_cuda_memory_bytes', 'device', 'gpu_name']
    for filename, records in [('resources.csv', [{k:r[k] for k in resource_fields} for r in results]), ('paired_differences.csv', pairs)]:
        if records:
            with (root/filename).open('w', newline='') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    state = '已完成' if complete else {'stopped':'已暂停', 'failed':'异常退出'}.get(active['phase'], '进行中')
    report = ['# GraphLoRA-adapted joint-QAT：MNIST GIN 对照', '',
        f"{state}；训练 {len(results)}/{expected}；最终测试 {tests}/{expected if study['profile']=='full' else 0}。", '',
        '**本轮是用户确认的同域图分类移植**：保留并行低秩图分支、监督对比目标和图内结构约束；'
        '没有跨图预训练源域，因此不启用特征投影或 SMMD；骨干和可训练量化参数联合更新。', '',
        '| 模式 | 变体 | 完成组数 | 最低验证 CE 检查点的验证准确率 % | 最终测试准确率 % | 适配参数 KiB |',
        '|---|---|---|---|---|---|']
    for row in rows:
        report.append(f"| {row['mode']} | {row['variant']} | {row['fits']}/{row['seeds']} | "
            f"{show(row['validation_accuracy'])} | {show(row['test_accuracy'])} | {show(row['adapter_KiB'])} |")
    report += ['', '均值 ± 样本标准差；单种子只显示该值及 n=1。每组严格按最低验证交叉熵选模，平局取最早轮次。', '']
    for config in study['configs']:
        split = '128/64/64 冒烟子集' if study['profile']=='smoke' else '55,000/5,000/10,000 官方完整划分'
        report += [f"- {config['quantization_mode']}：{split}，{config['epochs']} 轮，种子 {config['seeds']}，rank={config['graphlora_rank']}；"
            f"对比系数 {config['contrast_weight']}、结构系数 {config['structure_weight']}，tau={config['contrast_tau']}，同类正样本权重 {config['contrast_supervised_weight']}。"]
    report += ['', '共同骨干初始化、训练划分、数据顺序、基础优化器分组和学习率与同期无适配对照配对；'
        '全部输入 GPF 和聚合 LoRAP 移除。FP32 图分支只使用骨干已经量化一次的消息张量，原量化器及其预算不重复累计。', '',
        '图分类扩展：对比对象为末层主分支与合成分支的池化图表示；监督信号仅用训练图标签，'
        '不把数字类别当作超像素节点标签。结构目标按每张图的邻接计算，排除自配对、跨图配对及 padding。'
        '节点概率在训练时复用分类头 FP32 主权重，不增加分类参数或再次调用量化器。', '',
        'GIN 低秩分支包含两层低秩线性变换、中间 BatchNorm 和自环系数；只有末层 up 因子置零，'
        '末端保留有符号输出，使初始残差为零且可获得梯度。此初始化及图级损失是对官方 GAT 节点迁移实现的明确改动。', '',
        '额外适配参数与 BatchNorm 缓冲区分别统计。rank=2 时为 4,404 个参数（17.20 KiB），'
        '另有 3,552 字节缓冲区。实际位宽为量化骨干的元素加权有效激活位宽，不含 FP32 图分支中间激活；'
        '原 Table V 位宽定义仍需对齐，不能把本轮不同定义混在同一列。训练耗时和峰值显存见 resources.csv。', '',
        '官方来源：[GraphLoRA 论文](https://arxiv.org/abs/2409.16670)、'
        '[固定提交源码](https://github.com/AllminerLab/GraphLoRA/tree/aa9ab3dec32cee143c27fdbf07806d33356690db)。'
        '完整实现差异与复现命令见源码快照中的 GRAPHLORA_METHOD.md。']
    if study['profile']!='full':
        report += ['', '**本轮为预实验，仅使用训练集和验证集，最终测试次数为 0；这些数字不能直接填入 Table V 的测试准确率列。**']
    else:
        report += ['', '所有模式和种子的预定训练完成后才统一释放测试，每个检查点只测试一次。']
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    tex = ['% GraphLoRA-adapted joint-QAT. Align the bit definition before inserting into Table V.',
           '% Pilot validation metrics are deliberately excluded from test accuracy rows.']
    for row in rows:
        if row['variant']!='graphlora':continue
        ready = study['profile']=='full' and row['test_accuracy'] is not None
        bits = show(row['effective_bits'], 4).split(' (n=')[0] if ready else 'TBD'
        storage = show(row['adapter_KiB']).split(' (n=')[0] if ready else 'TBD'
        accuracy = show(row['test_accuracy']).split(' (n=')[0] if ready else 'TBD'
        tex += [f"% {row['mode']}", r'$+\,\mathrm{GraphLoRA\text{-}adapted}$ & $'+bits.replace('±',r'\pm')+
                r'$ & $'+storage.replace('±',r'\pm')+r'\,\mathrm{KiB}$ & $'+accuracy.replace('±',r'\pm')+r'$ \\']
    (root/'table_v_graphlora_rows.tex').write_text('\n'.join(tex)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    generate_report(parser.parse_args().root)
