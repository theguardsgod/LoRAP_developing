"""Read-only artifact audit and Chinese report. No model or test-set execution."""

import argparse
import csv
import json
from pathlib import Path
import shutil
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..residual_channel.experiment import sha
from ..residual_adapt.run import verify_selection
from ..run import write_json
from .config import ROOT,REPO,DATASETS,MODELS,SEEDS,VARIANTS


def load(path): return json.loads(path.read_text())


def stat(values):
    values=[float(v) for v in values]
    mean=statistics.mean(values);sd=statistics.stdev(values)
    half=4.302652729911275*sd/len(values)**.5
    return dict(values=values,mean=mean,sd=sd,ci95=[mean-half,mean+half])


def local_mean(rows,key):
    return [statistics.mean(r[key][i]['mse_reduction'] for r in rows) for i in range(len(rows[0][key]))]


def percent_list(values): return ' / '.join(f'{100*v:+.2f}' for v in values)


def inspect(root):
    release=load(root/'test_release.json');records=[]
    verification=dict(combinations=27,adaptations=81,epochs=100,reused_baselines=27,
        gpu_unit_tests=11,new_test_conditions=81,source_files_checked=0,
        export_audits=0,baseline_and_export_checks=0,common_reference_comparisons=0,zero_epoch_selections=0,last_epoch_selections=0,
        max_parameter_gap=0.,test_release_before_all_new_tests=True)
    for name,digest in load(root/'source_manifest.json').items():
        assert sha(root/'source_snapshot'/name)==digest
        verification['source_files_checked']+=1
    tests=load(root/'tests_passed.json')
    assert tests['count']==11 and tests['log_sha256']==sha(root/'tests.log')
    assert tests['manifest_sha256']==sha(root/'source_manifest.json')
    checked_sources=set()
    for d in DATASETS:
        for m in MODELS:
            for s in SEEDS:
                path=root/d/m/f'seed_{s}';r=load(path/'result.json')
                assert r['dataset']==d and r['architecture']==m and r['seed']==s
                assert r['test_metric_evaluations']==3 and r['device']=='cuda:0'
                assert r['config']['epochs']==100 and r['config']['error_loss_weight']==.1
                for name,digest in release['fits'][str(path.relative_to(root))].items(): assert sha(path/name)==digest
                assert load(path/'test_started.json')['started_unix']>= (root/'test_release.json').stat().st_mtime
                source=Path(r['base']['path'])
                for name,digest in r['base']['hashes'].items():
                    if str(source/name) not in checked_sources:
                        assert sha(source/name)==digest;checked_sources.add(str(source/name))
                split=load(path/'split.json')
                assert split==load(source/'split.json')
                training_ids=set(split['indices']['train'])
                calibration=load(path/'calibration.json')
                assert all(set(layer['training_node_ids'])==training_ids for layer in calibration)
                backbone=torch.load(source/'baseline/best.pt',map_location='cpu',weights_only=False)['model']
                assert r['variants']['task']['initial_prompt_sha256']==r['variants']['error']['initial_prompt_sha256']
                reference=r['variants']['task']['fixed_baseline_input_validation']
                for variant in ('error','mlp'):
                    for a,z in zip(reference,r['variants'][variant]['fixed_baseline_input_validation']):
                        assert a['before_sq']==z['before_sq']
                        assert a['squared_singular_values']==z['squared_singular_values']
                        verification['common_reference_comparisons']+=1
                for v in VARIANTS:
                    row=r['variants'][v]
                    best=torch.load(path/v/'best.pt',map_location='cpu',weights_only=False)
                    last=torch.load(path/v/'last.pt',map_location='cpu',weights_only=False)
                    verify_selection(last,best)
                    assert last['epoch']==100 and best['epoch']==row['best_epoch']
                    assert row['validation']['loss']==best['validation_loss']
                    assert last['history'][0]['validation']==r['base']['validation']
                    assert load(path/v/'test.json')==row['test']
                    exported=torch.load(path/v/'inference.pt',map_location='cpu',weights_only=False)
                    assert exported['backbone'].keys()==backbone.keys()
                    assert all(torch.equal(value,exported['backbone'][name]) for name,value in backbone.items())
                    assert exported['predictors'].keys()==best['model'].keys()
                    assert all(torch.equal(value,exported['predictors'][name]) for name,value in best['model'].items())
                    assert 'training_loss_scales' not in exported
                    assert all(row['inference_audit'].values()) and row['diagnostic_switch_exact']
                    assert row['backbone_unchanged'] and row['normalization_unchanged']
                    assert row['extra_residual_message_bits']==0
                    assert all(layer['relative_gap']<=.05 for layer in row['layers'])
                    verification['max_parameter_gap']=max(verification['max_parameter_gap'],*[layer['relative_gap'] for layer in row['layers']])
                    verification['export_audits']+=1;verification['baseline_and_export_checks']+=1
                    verification['zero_epoch_selections']+=best['epoch']==0
                    verification['last_epoch_selections']+=best['epoch']==100
                    for key in ('local_validation','local_training','fixed_baseline_input_validation'):
                        for layer in row[key]:
                            before,after,dot,norm=[layer[x] for x in ('before_sq','after_sq','dot','correction_sq')]
                            assert abs(after-before+2*dot-norm)/max(before,1e-30)<1e-8
                            assert layer['mse_reduction'] is None if before==0 else abs(layer['mse_reduction']-(1-after/before))<1e-12
                records.append(r)
    verification['source_artifacts_checked']=len(checked_sources)
    verification.update(frozen_backbones_verified=True,teacher_free_exports_verified=True,
        paired_initialization_verified=True,training_only_calibration_verified=True,
        minimum_validation_ce_selection_verified=True,local_error_identity_verified=True)
    write_json(root/'verification.json',verification)
    return records,verification


def analyze(records):
    groups=[]
    for d in DATASETS:
        for m in MODELS:
            rs=[r for r in records if r['dataset']==d and r['architecture']==m]
            group=dict(dataset=d,architecture=m,baseline=stat([r['base']['test']['accuracy']*100 for r in rs]),variants={})
            for v in VARIANTS:
                vr=[r['variants'][v] for r in rs]
                group['variants'][v]=dict(test=stat([r['test']['accuracy']*100 for r in vr]),
                    difference_vs_baseline=stat([(r['variants'][v]['test']['accuracy']-r['base']['test']['accuracy'])*100 for r in rs]),
                    difference_vs_task=stat([(r['variants'][v]['test']['accuracy']-r['variants']['task']['test']['accuracy'])*100 for r in rs]),
                    selected_epochs=[r['best_epoch'] for r in vr],parameters=vr[0]['parameters'],buffer_bytes=vr[0]['buffer_bytes'],
                    local_training=local_mean(vr,'local_training'),local_validation=local_mean(vr,'local_validation'),
                    fixed_baseline_input_validation=local_mean(vr,'fixed_baseline_input_validation'),
                    fixed_cosine=[statistics.mean(r['fixed_baseline_input_validation'][i]['cosine'] or 0. for r in vr)
                        for i in range(len(vr[0]['fixed_baseline_input_validation']))],
                    fixed_shuffled=[statistics.mean(r['fixed_baseline_input_validation'][i]['shuffled_mse_reduction'] for r in vr)
                        for i in range(len(vr[0]['fixed_baseline_input_validation']))])
            groups.append(group)
    return groups


def plots(root,groups,records):
    folder=root/'plots';folder.mkdir(exist_ok=True)
    colors=dict(none='#666666',task='#377eb8',error='#e66101',mlp='#4daf4a')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
    def save(fig,name):
        for extension in ('png','pdf','svg'): fig.savefig(folder/f'{name}.{extension}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(14,4.4),layout='constrained')
    for ax,d in zip(axes,DATASETS):
        gs=[g for g in groups if g['dataset']==d];x=np.arange(3)
        for j,v in enumerate(('none',)+VARIANTS):
            values=[g['baseline'] if v=='none' else g['variants'][v]['test'] for g in gs]
            ax.bar(x+(j-1.5)*.19,[a['mean'] for a in values],width=.18,color=colors[v],label=v,
                yerr=[a['sd'] for a in values],capsize=2)
        ax.set_xticks(x,MODELS);ax.set_ylim(0,100);ax.set_title(d);ax.set_ylabel('Test accuracy (%)')
    axes[-1].legend(ncol=2,loc='lower right');fig.suptitle('Quantized-input-only predictors: mean +/- seed SD (3 seeds)')
    save(fig,'test_accuracy')
    fig,ax=plt.subplots(figsize=(10,5.2),layout='constrained')
    x=np.arange(len(groups))
    for shift,v in ((-.12,'task'),(.12,'error')):
        values=[g['variants'][v]['difference_vs_baseline'] for g in groups]
        means=np.array([r['mean'] for r in values]);half=np.array([r['ci95'][1]-r['mean'] for r in values])
        ax.errorbar(x+shift,means,yerr=half,fmt='o',capsize=3,color=colors[v],label=v)
    ax.axhline(0,color='gray',ls='--');ax.set_xticks(x,[g['dataset']+'\n'+g['architecture'] for g in groups])
    ax.set_ylabel('Paired accuracy change vs none (percentage points)');ax.legend()
    ax.set_title('Mean and unadjusted paired t 95% interval; n=3 per group')
    save(fig,'paired_differences')
    fig,axes=plt.subplots(3,3,figsize=(13,10),layout='constrained')
    for ax,g in zip(axes.flatten(),groups):
        rs=[r for r in records if r['dataset']==g['dataset'] and r['architecture']==g['architecture']]
        # Keep physical layers separate with line style, and normalize each
        # layer by its common no-prompt reference energy before seed averaging.
        layers=len(rs[0]['variants']['error']['fixed_baseline_input_validation'])
        for i in range(layers):
            style='-' if i==0 else '--'
            physical=rs[0]['variants']['error']['fixed_baseline_input_validation'][i]['layer']
            for v in ('none',)+VARIANTS:
                spectra=[]
                for r in rs:
                    row=r['variants']['error' if v=='none' else v]['fixed_baseline_input_validation'][i]
                    values=row['squared_singular_values'] if v=='none' else row['after_squared_singular_values']
                    spectra.append(np.asarray(values)/max(row['before_sq'],1e-30))
                values=np.mean(spectra,axis=0)
                ax.plot(np.arange(1,len(values)+1),np.maximum(values,1e-14),ls=style,color=colors[v],label=f'{v} L{physical}')
        ax.set_yscale('log');ax.set_title(g['dataset']+' '+g['architecture'])
        ax.set_xlabel('Singular component');ax.set_ylabel('Energy / reference total')
        ax.legend(fontsize=7,ncol=2)
    fig.suptitle('Validation error spectra on identical baseline inputs (solid: first block; dashed: second)')
    save(fig,'fixed_input_error_spectra')


def report(root,groups,records,verification):
    lines=['# 仅量化信息提示：误差监督首轮 GPU 报告','',
        '已完成 3 数据集 × 3 模型 × 3 种子，共 27 组、81 次适配，每次 100 轮。27 个无补偿骨干全部复用，冻结骨干与量化参数。',
        '本轮推理提示不读取量化前消息，训练阶段以局部量化误差监督。全部 81 个导出模型通过禁用参考函数、毒化量化前回调参数、删除输入标签后的输出精确等价核对。',
        '这是固定骨干的量化条件低秩提示实验，不是原论文 LoRAP 全量训练复现；旧的真实残差通道也不作为本轮可部署方案。','',
        '**主要发现：误差监督在 14/15 个层级均值上比同容量 task 更接近局部目标，但只有 9/15 个实际降低了相对无提示的误差；平均准确率只在 2/9 个组合优于 task。推理信息约束已满足，稳定任务收益尚未证实。**','',
        '## 方法和对照','',
        '- none：同一个既有无补偿最佳骨干；测试指标直接引用上一轮文件。',
        '- task：量化条件低秩提示，只训练分类 CE。',
        '- error：与 task 完全相同结构和初始参数，额外加入 0.1 × 逐层归一化误差 MSE 的平均值。',
        '- mlp：读取相同量化特征与条件的两层 ReLU 残差 MLP，参数数目逐层与低秩提示相差不超过 5%，只训练 CE。','',
        'rank=2、num_prompts=20，每个量化块独立参数；GAT 第一层在拼接后的 64 维上总 rank=2。三种适配器初始输出均为零。',
        '每次适配 100 轮，Adam 主学习率为 0.01（GAT-CiteSeer/PubMed 为 0.005），零权重衰减。选择最低验证 CE，包含 epoch 0，平局保留较早轮次。',
        '步长、舍入位宽、入度和边界码占比作为条件。边界码占比不是实际裁剪率。全部推理标准化统计只取训练节点的量化数据；训练误差归一化单独保存，不进入部署导出。','',
        '## 测试准确率','',
        '单位为 %，显示三个种子的均值 ± 样本标准差。差值单位为百分点。所有 81 次适配锁定后才计算本轮新增测试指标。','',
        '| 数据集 | 模型 | none | task | error | mlp | error−none | error−task |',
        '|---|---|---:|---:|---:|---:|---:|---:|']
    for g in groups:
        def value(v):
            s=g['baseline'] if v=='none' else g['variants'][v]['test']
            return f"{s['mean']:.2f} ± {s['sd']:.2f}"
        e=g['variants']['error']
        lines.append(f"| {g['dataset']} | {g['architecture']} | "+' | '.join(value(v) for v in ('none',)+VARIANTS)+
            f" | {e['difference_vs_baseline']['mean']:+.2f} | {e['difference_vs_task']['mean']:+.2f} |")
    lines += ['', '![测试准确率](plots/test_accuracy.png)','',
        '## 配对差值的不确定性','',
        '仅 3 个种子，使用 df=2 的配对 t 区间；未校正九个组合的多重比较，区间不能当作充分的显著性证明。','',
        '| 数据集 | 模型 | error−none：均值 [95% 区间] | error−task：均值 [95% 区间] |',
        '|---|---|---:|---:|']
    for g in groups:
        e=g['variants']['error']
        def diff(key):
            v=e[key];return f"{v['mean']:+.2f} [{v['ci95'][0]:+.2f}, {v['ci95'][1]:+.2f}]"
        lines.append(f"| {g['dataset']} | {g['architecture']} | {diff('difference_vs_baseline')} | {diff('difference_vs_task')} |")
    lines += ['', '![配对差值](plots/paired_differences.png)','',
        '## 固定输入的验证误差：是否学到补偿','',
        '所有适配器都在同一个无提示骨干的量化输入上单独生成提示；不把上游提示改变误差大小混入对照。',
        '下表为 100×(1−补偿后平方误差/补偿前平方误差)，负数表示误差增大。GCN/GAT 两个数按物理层 1 / 2；GIN 只有物理层 2。各数为三种子的平均比例。','',
        '| 数据集 | 模型 | task | error | mlp | error 提示置换后 | error 余弦相似度 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for g in groups:
        rows=g['variants'];e=rows['error']
        lines.append(f"| {g['dataset']} | {g['architecture']} | "+
            ' | '.join(percent_list(rows[v]['fixed_baseline_input_validation']) for v in VARIANTS)+
            ' | '+percent_list(e['fixed_shuffled'])+' | '+' / '.join(f'{v:+.3f}' for v in e['fixed_cosine'])+' |')
    lines += ['', '置换在验证节点内进行，保持提示向量集合与幅度分布，每层固定 20 个随机排列。原提示优于置换支持节点对应关系有用；若两者相近，可能主要学到全局偏置。',
        '这些诊断衡量固定当前算子下的局部值消息量化误差，GAT 固定量化注意力；它们不是完整 FP32 模型的误差。','',
        '误差下降比例是相对各自基线误差的比值；当原误差很小时，提示偏移会得到很大的负百分比。绝对平方误差和每种子分解保存在 result.json。比 task 更少的误差放大不能等同于相对无提示的成功补偿。','',
        '![固定输入误差谱](plots/fixed_input_error_spectra.png)','',
        '## 当前适配模型输入上的误差','',
        '用于查看多层提示共同作用后的实际局部行为；不同变体输入不再相同，不能单靠此表作补偿能力的公平比较。','',
        '| 数据集 | 模型 | task 验证 | error 验证 | mlp 验证 | error 训练 |',
        '|---|---|---:|---:|---:|---:|']
    for g in groups:
        rows=g['variants']
        lines.append(f"| {g['dataset']} | {g['architecture']} | "+
            ' | '.join(percent_list(rows[v]['local_validation']) for v in VARIANTS)+
            ' | '+percent_list(rows['error']['local_training'])+' |')
    lines += ['', '## 参数开销与选模','',
        '参数均为 FP32。冻结骨干及其位宽与来源检查点完全相同；提示不额外传输残差码，但增加提示计算及下列状态。',
        '不能将“没有残差通信”解读成零额外计算，也没有进行端到端整数部署或速度测量。','',
        '| 数据集 | 模型 | task/error 参数 | mlp 参数 | 固定推理统计字节 | error 最佳轮次（41/42/43） |',
        '|---|---|---:|---:|---:|---|']
    for g in groups:
        e=g['variants']['error'];mlp=g['variants']['mlp']
        lines.append(f"| {g['dataset']} | {g['architecture']} | {e['parameters']} | {mlp['parameters']} | {e['buffer_bytes']} | "+
            ' / '.join(map(str,e['selected_epochs']))+' |')
    lines += ['',f"全部适配中，{verification['zero_epoch_selections']}/81 选择 epoch 0，不能计作有效补偿；{verification['last_epoch_selections']}/81 选择最后一轮，不保证已充分收敛。",
        f"逐层参数差异最大 {100*verification['max_parameter_gap']:.2f}%。检查点文件还包含序列化元数据，文件大小不等于逻辑参数大小。",'',
        '## 本轮能够支持的结论','']
    better=sum(g['variants']['error']['difference_vs_task']['mean']>1e-9 for g in groups)
    base_better=sum(g['variants']['error']['difference_vs_baseline']['mean']>1e-9 for g in groups)
    total_layers=sum(len(g['variants']['error']['fixed_baseline_input_validation']) for g in groups)
    positive_layers=sum(v>0 for g in groups for v in g['variants']['error']['fixed_baseline_input_validation'])
    better_layers=sum(e>t for g in groups for e,t in zip(g['variants']['error']['fixed_baseline_input_validation'],g['variants']['task']['fixed_baseline_input_validation']))
    lines += [f'- error 在 {better}/9 个组合的平均测试准确率高于 task，在 {base_better}/9 个组合高于无提示基线。',
        f'- 固定输入诊断的 {total_layers} 个“数据集×模型×层”均值中，error 有 {positive_layers} 个降低局部误差，有 {better_layers} 个优于 task 的误差下降比例。',
        '- 预测所需信息已符合本轮部署约束：提示只读取量化消息及元数据，训练参考不进入推理；这不代表误差可以精确恢复。',
        '- 只有同时比较固定输入误差、同容量 task、参数匹配 MLP 和分类指标，才能区分补偿证据与任务适配。局部误差下降不自动证明准确率增益由补偿造成。',
        '- 本轮只检验固定骨干、固定损失权重 0.1、rank 2、100 轮的设定；不对其他损失权重或联合训练效果作结论。',
        '- Cora/CiteSeer GIN 沿用较弱的历史基线；公共基准已多次用于研究。3 个种子及本轮结果不足以宣称论文级普遍收益。','',
        '## 验证和复现','',
        f"11 项 GPU 单元测试通过；81 个导出模型通过推理参考隔离/无标签/重载等价；{verification['source_files_checked']} 个训练源码文件和 {verification['source_artifacts_checked']} 个来源文件的哈希已核对。",
        '全部最佳检查点经逐轮验证损失独立核查；导出骨干张量逐项等于来源，导出提示等于所选最佳参数；误差恒等式与参数匹配均通过。',
        f"固定输入诊断还通过 {verification['common_reference_comparisons']} 个跨变体参考平方误差和误差谱的精确相等核对。",
        '详见 [verification.json](verification.json)、[逐种子表](per_seed.csv)、[完整分析](analysis.json)、[协议](METHOD.md)。','',
        '```sh',
        'PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.quant_predict.suite --gpus 0 1 2 3',
        'PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.quant_predict.report',
        '```','',
        '训练与测试以 source_snapshot 内不可变源码执行。该报告由另存的 analysis_snapshot 生成，不进行模型前向或重新计算测试指标。',
        '三种模型的 2 轮流程检查保存在独立 smoke 目录，不纳入正式统计。原始 LoRAP、残差通道、历次结果和用户修改均保留。']
    (root/'REPORT.md').write_text('\n'.join(lines)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    args=parser.parse_args();root=args.root.resolve()
    analysis=root/'analysis_snapshot';analysis.mkdir(exist_ok=True)
    manifest={}
    for path in (Path(__file__).resolve(),Path(__file__).resolve().with_name('config.py')):
        target=analysis/path.name
        if target.exists(): assert sha(path)==sha(target),'Preserve prior analysis before changing it'
        else: shutil.copy2(path,target)
        manifest[path.name]=sha(path)
    write_json(root/'analysis_source_manifest.json',manifest)
    records,verification=inspect(root)
    groups=analyze(records)
    write_json(root/'analysis.json',groups)
    with (root/'per_seed.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=['dataset','model','seed','variant','test_accuracy_pct','baseline_accuracy_pct','difference_pp',
            'best_epoch','parameters','fixed_input_error_reduction_pct','current_input_error_reduction_pct'])
        writer.writeheader()
        for r in records:
            for v in VARIANTS:
                row=r['variants'][v]
                writer.writerow(dict(dataset=r['dataset'],model=r['architecture'],seed=r['seed'],variant=v,
                    test_accuracy_pct=row['test']['accuracy']*100,baseline_accuracy_pct=r['base']['test']['accuracy']*100,
                    difference_pp=(row['test']['accuracy']-r['base']['test']['accuracy'])*100,best_epoch=row['best_epoch'],parameters=row['parameters'],
                    fixed_input_error_reduction_pct=percent_list([a['mse_reduction'] for a in row['fixed_baseline_input_validation']]),
                    current_input_error_reduction_pct=percent_list([a['mse_reduction'] for a in row['local_validation']])))
    plots(root,groups,records);report(root,groups,records,verification)
    print(json.dumps(verification,indent=2))
    print(root/'REPORT.md')


if __name__=='__main__': main()
