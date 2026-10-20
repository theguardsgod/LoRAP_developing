"""Summarize GLoRA joint-QAT runs and archived matched controls, without inference."""
import argparse
import csv
import json
from pathlib import Path
import statistics

from .run import write_json


def read(path):
    return json.loads(path.read_text())


def stats(values):
    return None if not values else dict(mean=statistics.mean(values),sample_std=statistics.stdev(values) if len(values)>1 else None,n=len(values))


def show(value,precision=2):
    if value is None:return '待测'
    if value['sample_std'] is None:return f"{value['mean']:.{precision}f} (n=1)"
    return f"{value['mean']:.{precision}f} ± {value['sample_std']:.{precision}f}"


def generate_report(root):
    root=Path(root);study=read(root/'study.json')
    rows,pairs,resources,progress=[],[],[],[]
    trained=searched=tested=expected=0
    for config in study['configs']:
        mode=config['quantization_mode'];results=[];references=[];loras=[]
        expected+=len(config['seeds'])
        for seed in config['seeds']:
            directory=root/mode/'MNIST/glora'/f'seed_{seed}'
            reference=read(root/'reference'/mode/f'seed_{seed}'/'result.json')
            references.append(reference);loras.append(read(root/'reference'/mode/f'seed_{seed}'/'lora_result.json'))
            if (directory/'fit.json').exists():trained+=1
            if (directory/'progress.json').exists():progress.append(dict(mode=mode,seed=seed,**read(directory/'progress.json')))
            if not (directory/'result.json').exists():continue
            result=read(directory/'result.json');results.append(result);searched+=1
            assert result['common_backbone_sha256']==reference['common_backbone_sha256']
            assert result['compact_counts']['external_prompt_parameters']==0
            resources.append(dict(mode=mode,seed=seed,supernet_trainable_parameters=result['supernet_counts']['trainable_parameters'],
                supernet_adapter_parameters=result['supernet_counts']['adapter_parameters'],
                selected_adapter_parameters=result['active_adapter_parameters'],selected_adapter_bytes=result['active_adapter_fp32_bytes'],
                compact_total_parameters=result['compact_counts']['total_parameters'],
                activation_effective_bits=result['mean_activation_effective_bits'],training_seconds=result['training_seconds'],
                search_validation_seconds=result['search_validation_seconds'],search_candidates=result['search_evaluations'],
                peak_training_cuda_bytes=result['peak_training_cuda_bytes'],peak_search_cuda_bytes=result['peak_search_cuda_bytes'],device=result['device']))
            if result['test_evaluations']:
                tested+=1
                pairs.append(dict(mode=mode,seed=seed,glora_accuracy_percent=100*result['test']['accuracy'],
                    no_prompt_accuracy_percent=100*reference['test']['accuracy'],lora_accuracy_percent=100*loras[-1]['test']['accuracy'],
                    glora_minus_no_prompt_pp=100*(result['test']['accuracy']-reference['test']['accuracy']),
                    glora_minus_lora_pp=100*(result['test']['accuracy']-loras[-1]['test']['accuracy'])))
        mode_pairs=[p for p in pairs if p['mode']==mode]
        ready=len(mode_pairs)==len(config['seeds']) and not study['smoke']
        rows.append(dict(mode=mode,complete=ready,fits=len(results),tests=len(mode_pairs),
            no_prompt_accuracy=stats([100*r['test']['accuracy'] for r in references]) if not study['smoke'] else None,
            lora_accuracy=stats([100*r['test']['accuracy'] for r in loras]) if not study['smoke'] else None,
            glora_accuracy=stats([p['glora_accuracy_percent'] for p in mode_pairs]) if ready else None,
            glora_minus_no_prompt_pp=stats([p['glora_minus_no_prompt_pp'] for p in mode_pairs]) if ready else None,
            glora_minus_lora_pp=stats([p['glora_minus_lora_pp'] for p in mode_pairs]) if ready else None,
            active_adapter_parameters=stats([r['active_adapter_parameters'] for r in results]),
            active_adapter_KiB=stats([r['active_adapter_fp32_bytes']/1024 for r in results]),
            mean_activation_effective_bits=stats([r['mean_activation_effective_bits'] for r in results])))
    complete=searched==expected and (study['smoke'] or tested==expected)
    write_json(root/'status.json',dict(complete=complete,smoke=study['smoke'],supernets_completed=trained,searches_completed=searched,
        fits_total=expected,tests_completed=tested,tests_expected=0 if study['smoke'] else expected,rows=rows,progress=progress))
    write_json(root/'comparison.json',rows)
    for filename,records in [('paired_differences.csv',pairs),('resources.csv',resources)]:
        if records:
            with (root/filename).open('w',newline='') as stream:
                writer=csv.DictWriter(stream,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    report=['# GLoRA（One-for-All）joint-QAT baseline','',
        f"超网训练 {trained}/{expected}；搜索与导出 {searched}/{expected}；最终测试 {tested}/{0 if study['smoke'] else expected}；{'已完成' if complete else '进行中'}。",'',
        '方法为 GLoRA 五支持张量（A/B/C/D/E）超网及逐层进化结构搜索，官方源代码固定到 '
        '`Arnav0400/ViT-Slim@390467ddf529d747168709ce2f007a85a0399cca`；许可证与原始文件随快照保存。','',
        '量化线性层采用 `Linear(x_q, W_q + W_q ⊙ A + B, b + b ⊙ D + E + W_q C)`。'
        'A/B 有低秩、向量、标量、关闭四种选项；C 有低秩、向量、关闭三种；D/E 有标量、关闭、向量三种。'
        '每层共 432 种结构，训练时独立随机抽样；每个量化器仍只调用一次，特征预算仅累计一次。','',
        '| 协议 | 无适配测试准确率 % | 标准 LoRA % | GLoRA % | GLoRA − 无适配（百分点） | GLoRA − LoRA（百分点） |',
        '|---|---|---|---|---|---|']
    for row in rows:
        report.append(f"| {row['mode']} | {show(row['no_prompt_accuracy'])} | {show(row['lora_accuracy'])} | {show(row['glora_accuracy'])} | {show(row['glora_minus_no_prompt_pp'])} | {show(row['glora_minus_lora_pp'])} |")
    report += ['','均值 ± 样本标准差；逐种子配对差值见 paired_differences.csv。','',
        '| 协议 | 搜索后适配参数数 | 搜索后 FP32 KiB | 有效激活位宽 |','|---|---|---|---|']
    for row in rows:
        report.append(f"| {row['mode']} | {show(row['active_adapter_parameters'])} | {show(row['active_adapter_KiB'])} | {show(row['mean_activation_effective_bits'],4)} |")
    report += ['','## 训练与原方法的区别','',
        '本轮按用户确认的 Table V 协议联合训练 GIN 骨干与既有可训练量化参数；原 GLoRA 视觉实验冻结预训练骨干。'
        '因此应标注 **GLoRA joint-QAT**，不声称复现其 ViT/VTAB 论文准确率，也不标作 GraphLoRA。','',
        '骨干、数据划分、优化器基础分组、学习率和量化损失继承无提示对照；移除输入 GPF 与聚合 LoRAP。'
        '独立初始化和子网随机数流保持公共骨干及训练数据顺序配对。GLoRA 适配参数学习率等于骨干主学习率，权重衰减为 0。','']
    for config in study['configs']:
        split='128/64/64 冒烟子集' if study['smoke'] else '55,000/5,000/10,000 官方完整划分'
        report.append(f"- {config['quantization_mode']}：{split}，{config['epochs']} 轮，种子 {config['seeds']}，rank={config['glora_rank']}；"
                      f"搜索 {config['search_population']} 个个体 × {config['search_generations']} 代，精英 {config['search_elite']}，"
                      f"逐路径变异概率 {config['search_mutation']}，适配参数上限 {config['max_adapter_parameters']}。")
    report += ['','选模分两步：先以固定全低秩 A/B/C、向量 D/E 结构的最低验证 CE 选择超网检查点，平局取较早轮次；'
        '再仅在验证集上进化搜索该检查点的结构，按最低 CE 选结构，平局取较早候选。搜索包含全路径和全关闭两个初始候选。'
        'BatchNorm 统计在搜索和评估时冻结，使用超网训练所得统计。','',
        '相比原论文的 500 轮超网训练和验证准确率搜索，本轮使用既有训练轮数和验证 CE。'
        '官方 ViT 实现不向分类头注入支持张量；本轮覆盖全部量化线性层，包含分类头。'
        '相比现有无适配/LoRA 对照，GLoRA 增加了结构搜索及验证访问；搜索耗时单列，不能视为等搜索预算实验。','',
        '所有预定超网训练与搜索完成后才释放最终测试，每个压缩模型只测试一次。现有对照仅读取归档结果，不重复测试。'
        'FP32 支持张量不合并后重新量化，不声称量化部署零开销；导出会真正删除所选路径不使用的超网参数。'
        'rank=2 时超网有 13,300 个适配参数（51.95 KiB），所选模型参数量以实际导出为准。','',
        '实际位宽按验证集中经过量化器的激活元素加权，FP32 适配开销另列；与原论文表格的 Average bits 定义仍需另行核对。'
        '资源统计及耗时见 resources.csv；search_validation_seconds 为所有已执行候选验证的累计耗时，含中断后重复验证成本，'
        '不含检查点读取及最终导出验证。恢复命令见 RUN_COMMANDS.md。','',
        '方法来源：[GLoRA 论文](https://arxiv.org/abs/2306.07967)、'
        '[官方实现](https://github.com/Arnav0400/ViT-Slim/tree/390467ddf529d747168709ce2f007a85a0399cca/GLoRA)。']
    if study['smoke']:report += ['','本次只验证真实 GPU 数据流程，不评估测试集，不提供正式性能结论。']
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')
    tex=['% GLoRA joint-QAT; structures searched on validation only.',
         '% Active FP32 adapter KiB is mean +/- sample std across selected seed-specific structures.',
         '% Effective activation bits; do not substitute for another bit definition without aligning it.']
    for row in rows:
        bits=show(row['mean_activation_effective_bits'],4) if row['complete'] else 'TBD'
        storage=show(row['active_adapter_KiB']) if row['complete'] else 'TBD'
        accuracy=show(row['glora_accuracy']) if row['complete'] else 'TBD'
        tex += [f"% {row['mode']} block",r'$+\,\mathrm{GLoRA}$ & $'+bits.replace('±',r'\pm')+
                r'$ & $'+storage.replace('±',r'\pm')+r'\,\mathrm{KiB}$ & $'+accuracy.replace('±',r'\pm')+r'$ \\']
    (root/'table_v_glora_rows.tex').write_text('\n'.join(tex)+'\n')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('root',type=Path)
    generate_report(parser.parse_args().root)
