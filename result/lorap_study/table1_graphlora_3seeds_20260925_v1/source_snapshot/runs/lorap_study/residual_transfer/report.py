"""Artifact-only audit, paired summaries, error spectra and Chinese report."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import t
import torch

from ..residual_adapt.run import verify_selection
from ..residual_channel.experiment import sha
from ..run import write_json
from .config import ROOT,DATASETS,MODELS,SEEDS,MODES

LABELS=dict(none='无补偿基线',unit='固定全量补偿',gate='只训练门控',higher='同消息预算加位宽')
COLORS=dict(none='#727272',unit='#b75d76',gate='#287cb3',higher='#288c70')


def paired(values):
    a=np.asarray(values,float)
    mean=float(a.mean())
    radius=float(t.ppf(.975,len(a)-1)*a.std(ddof=1)/np.sqrt(len(a)))
    return dict(mean=mean,ci95=[mean-radius,mean+radius],wins=int((a>1e-10).sum()),
        ties=int((abs(a)<=1e-10).sum()),losses=int((a< -1e-10).sum()))


def analyze(root):
    root=root.resolve()
    release=json.loads((root/'test_release.json').read_text())
    assert release['all_gate_fits_complete'] and release['test_selections_frozen']
    for path,files in release['fits'].items():
        for name,digest in files.items():assert sha(root/path/name)==digest
    for kind in ('source','analysis_source'):
        snapshot=root/('source_snapshot' if kind=='source' else 'analysis_snapshot')
        for name,digest in json.loads((root/f'{kind}_manifest.json').read_text()).items():
            assert sha(snapshot/name)==digest
    assert 'Ran 12 tests' in (root/'tests.log').read_text() and '\nOK\n' in (root/'tests.log').read_text()
    transport=json.loads((root/'transport_audit.json').read_text())
    assert len(transport['conditions'])==108 and transport['test_metric_evaluations']==0 and transport['model_updates']==0
    assert all(r['logits_bitwise_equal'] and r['validation_metrics_exact'] for r in transport['conditions'])
    records,comparisons,flat=[],[],[]
    checked_sources={}
    histories={}
    zero_references=0
    for dataset in DATASETS:
        split_hashes=[]
        for architecture in MODELS:
            for seed in SEEDS:
                directory=root/dataset/architecture/f'seed_{seed}'
                r=json.loads((directory/'result.json').read_text())
                c=json.loads((directory/'config.json').read_text())
                split=json.loads((directory/'split.json').read_text())
                split_hashes.append(sha(directory/'split.json'))
                data_path=Path(c['data_root'])/dataset/'processed/data.pt'
                assert sha(data_path)==split['processed_sha256']
                assert c['trainable_quantizer_bit_bounds']==[1.,16.]
                assert r['gate_epochs']==100 and r['test_metric_evaluations']==4
                assert r['device'].startswith('cuda') and r['test_state_unchanged'] and r['source_unchanged']
                assert r['backbone_unchanged'] and r['diagnostic_switch_exact'] and r['bypass_side_exact']
                assert all(0<=v<=1 for v in r['gates'])
                best=torch.load(directory/'gate/best.pt',map_location='cpu',weights_only=False)
                last=torch.load(directory/'gate/last.pt',map_location='cpu',weights_only=False)
                base=torch.load(directory/'baseline/best.pt',map_location='cpu',weights_only=False)
                code=torch.load(directory/'calibration.pt',map_location='cpu',weights_only=False)
                verify_selection(last,best)
                assert last['epoch']==100 and best['epoch']==r['best_gate_epoch']
                assert best['model']['gates'].tolist()==r['gates']
                history=[json.loads(l) for l in (directory/'gate/epochs.jsonl').read_text().splitlines()]
                assert history==last['history']
                histories[(dataset,architecture,seed)]=history
                for name,value in base['model'].items():
                    assert torch.equal(value,best['model']['backbone.'+name])
                    assert torch.equal(value,last['model']['backbone.'+name])
                    if name.endswith('.bit'):
                        assert (value.abs().round()>=1).all(), (dataset,architecture,seed,name)
                for i,entry in enumerate(code):
                    for name,value in entry.items():
                        assert torch.equal(value,best['model'][f'codes.{i}.{name}'])
                        assert torch.equal(value,last['model'][f'codes.{i}.{name}'])
                calibration=json.loads((directory/'calibration.json').read_text())
                training=set(split['indices']['train'])
                for row,entry in zip(calibration,code):
                    assert set(row['basis_training_node_ids'])<=training
                    assert set(row['scale_training_node_ids'])<=training
                    assert len(row['basis_training_node_ids'])<=4096
                    assert int(entry['increments'].sum())==row['extra_message_bits_per_node']
                    assert row['head_orthogonality_error']<1e-5
                assert r['validation']['gate']['loss']==best['validation_loss']
                assert r['validation']['none']['loss']==r['base']['selected_validation']['loss']
                if not r['base']['reused']:
                    base_last=torch.load(directory/'baseline/last.pt',map_location='cpu',weights_only=False)
                    verify_selection(base_last,base)
                    assert base_last['epoch']==r['base']['epochs']==200
                for path,digest in r['base']['source_hashes'].items():
                    if path not in checked_sources:assert sha(path)==digest
                    else:assert checked_sources[path]==digest
                    checked_sources[path]=digest
                for mode in MODES:
                    metrics=r['test'][mode]
                    assert metrics==json.loads((directory/f'test_{mode}.json').read_text())
                    assert metrics['examples']==len(split['indices']['test'])
                    for layer in metrics['bit_audit']['layers']:
                        assert layer['main_bits_per_node']>=0 and layer['side_bits_per_node']>=0
                    for item in metrics['bit_audit']['other_activation_quantizers'].values():
                        assert min(map(float,item['histogram']))>=0
                        assert sum(item['histogram'].values())==item['elements']
                    for layer in r['validation'][mode]['local_diagnostics']:
                        before,after,dot,norm=(layer[k] for k in ('before_sq','after_sq','dot','correction_sq'))
                        assert abs(after-before+2*dot-norm)<=max(before,1e-30)*1e-6
                        zero_references+=before==0
                        if before>0:assert abs(layer['mse_reduction']-(1-after/before))<1e-12
                        else:assert layer['mse_reduction'] is None
                        if 'orthogonal_identity_relative_gap' in layer:
                            assert layer['orthogonal_identity_relative_gap']<2e-5
                    flat.append(dict(dataset=dataset,model=architecture,seed=seed,mode=mode,
                        test_accuracy=metrics['accuracy'],test_loss=metrics['loss'],
                        validation_accuracy=r['validation'][mode]['accuracy'],validation_loss=r['validation'][mode]['loss'],
                        best_gate_epoch=r['best_gate_epoch'],gates=json.dumps(r['gates'])))
                for a,b in zip(r['test']['gate']['bit_audit']['layers'],r['test']['higher']['bit_audit']['layers']):
                    assert abs(a['total_message_bits_per_node']-b['total_message_bits_per_node'])<1e-8
                if max(r['gates'])==0:
                    assert r['test']['gate']['accuracy']==r['test']['none']['accuracy']
                    assert r['test']['gate']['loss']==r['test']['none']['loss']
                records.append(r)
            group=[r for r in records if r['dataset']==dataset and r['architecture']==architecture]
            for method,reference in (('gate','none'),('gate','unit'),('gate','higher')):
                comparisons.append(dict(dataset=dataset,model=architecture,method=method,reference=reference,
                    test_accuracy_delta=paired([100*(r['test'][method]['accuracy']-r['test'][reference]['accuracy']) for r in group]),
                    validation_accuracy_delta=paired([100*(r['validation'][method]['accuracy']-r['validation'][reference]['accuracy']) for r in group])))
        assert len(set(split_hashes))==1
    assert len(records)==27 and len(flat)==108
    verification=dict(combinations=27,new_baselines=sum(not r['base']['reused'] for r in records),
        reused_baselines=sum(r['base']['reused'] for r in records),baseline_epochs=200,gate_epochs=100,
        formal_gpu_unit_tests=12,test_conditions=108,gpu_only=True,source_files_verified=len(checked_sources),
        source_and_analysis_snapshots_unchanged=True,frozen_backbones_verified=True,
        calibration_training_nodes_only=True,exact_additional_message_bit_match=True,
        local_squared_error_identity=True,valid_rounded_signed_bits=True,zero_error_references=zero_references,
        test_release_before_all_test_metrics=True,transport_equivalence_conditions=108,
        zero_gate_selections=sum(max(r['gates'])==0 for r in records),
        last_epoch_gate_selections=sum(r['best_gate_epoch']==100 for r in records))
    write_json(root/'verification.json',verification)
    write_json(root/'analysis.json',dict(records=records,comparisons=comparisons,verification=verification))
    with (root/'per_seed.csv').open('w',newline='') as file:
        writer=csv.DictWriter(file,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    def save(fig,name):
        for extension in ('png','pdf','svg'):
            fig.savefig(root/f'{name}.{extension}',dpi=180,bbox_inches='tight')
        plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12.8,4.2))
    for ax,dataset in zip(axes,DATASETS):
        for j,mode in enumerate(MODES):
            values=[[100*r['test'][mode]['accuracy'] for r in records if r['dataset']==dataset and r['architecture']==m] for m in MODELS]
            ax.errorbar(np.arange(3)+(j-1.5)*.15,np.mean(values,1),yerr=np.std(values,1,ddof=1),fmt='o',capsize=3,label=mode,color=COLORS[mode])
        ax.set_xticks(range(3),MODELS);ax.set_title(dataset);ax.grid(axis='y',alpha=.2)
    axes[0].set_ylabel('Test accuracy (%)');axes[0].legend(fontsize=8)
    fig.suptitle('Frozen residual gates: 3 seeds, mean ± SD');fig.tight_layout();save(fig,'test_comparison')

    fig,ax=plt.subplots(figsize=(10.5,4.2))
    selected=[c for c in comparisons if c['reference']=='none']
    for i,c in enumerate(selected):
        stat=c['test_accuracy_delta'];mean=stat['mean'];lo,hi=stat['ci95']
        ax.errorbar(i,mean,yerr=[[mean-lo],[hi-mean]],fmt='o',color=COLORS['gate'],capsize=4)
        deltas=[100*(r['test']['gate']['accuracy']-r['test']['none']['accuracy']) for r in records if r['dataset']==c['dataset'] and r['architecture']==c['model']]
        ax.scatter([i-.12,i,i+.12],deltas,color=COLORS['gate'],alpha=.5,s=16)
    ax.axhline(0,color='#555555',lw=.8);ax.set_xticks(range(9),[c['dataset']+' / '+c['model'] for c in selected],rotation=35,ha='right')
    ax.set_ylabel('Gate − baseline (percentage points)');ax.grid(axis='y',alpha=.2)
    ax.set_title('Paired test differences: 3 seeds, unadjusted 95% t intervals')
    fig.tight_layout();save(fig,'paired_differences')

    for dataset in DATASETS:
        fig,axes=plt.subplots(3,3,figsize=(11.5,7.5),sharex=True,sharey=True)
        for i,architecture in enumerate(MODELS):
            for j,seed in enumerate(SEEDS):
                h=histories[(dataset,architecture,seed)]
                for layer in range(len(h[0]['gates'])):
                    axes[i,j].plot([r['epoch'] for r in h],[r['gates'][layer] for r in h],label=f"Layer {layer+(2 if architecture=='GIN' else 1)}")
                r=next(r for r in records if (r['dataset'],r['architecture'],r['seed'])==(dataset,architecture,seed))
                axes[i,j].axvline(r['best_gate_epoch'],color='black',ls=':',lw=.8)
                axes[i,j].set_title(f'{architecture}, seed {seed}',fontsize=9);axes[i,j].set_ylim(-.02,1.02)
                axes[i,j].grid(alpha=.2)
        axes[1,0].legend(fontsize=8)
        for ax in axes[-1]:ax.set_xlabel('Epoch')
        for ax in axes[:,0]:ax.set_ylabel('Gate strength')
        fig.suptitle(dataset+': gate trajectories; dotted line = selected epoch');fig.tight_layout();save(fig,dataset+'_gates')

    fig,axes=plt.subplots(3,3,figsize=(11.5,8.2))
    for i,dataset in enumerate(DATASETS):
        for j,architecture in enumerate(MODELS):
            group=[r for r in records if r['dataset']==dataset and r['architecture']==architecture]
            for layer in range(len(group[0]['validation']['none']['local_diagnostics'])):
                values=np.array([r['validation']['none']['local_diagnostics'][layer]['cumulative_energy'] for r in group])
                axes[i,j].plot(np.arange(1,values.shape[1]+1),values.mean(0),label=f"Layer {layer+(2 if architecture=='GIN' else 1)}")
            axes[i,j].set_title(dataset+' / '+architecture,fontsize=9);axes[i,j].set_ylim(0,1.02)
            axes[i,j].grid(alpha=.2);axes[i,j].legend(fontsize=8)
    for ax in axes[-1]:ax.set_xlabel('Spectral rank')
    for ax in axes[:,0]:ax.set_ylabel('Cumulative squared-error energy')
    fig.suptitle('Local value-quantization error spectra on validation nodes');fig.tight_layout();save(fig,'error_spectra')

    def group(d,m):return [r for r in records if r['dataset']==d and r['architecture']==m]
    def meanstd(rows,mode,split='test'):
        a=np.array([100*r[split][mode]['accuracy'] for r in rows]);return f'{a.mean():.3f} ± {a.std(ddof=1):.3f}'
    findings=[]
    for architecture in MODELS:
        changes=[next(c['test_accuracy_delta']['mean'] for c in selected if c['dataset']==d and c['model']==architecture) for d in DATASETS]
        findings.append(architecture+'：'+', '.join(f'{v:+.3f}' for v in changes)+' 个百分点')
    text=['# 只训练门控：跨数据集与模型验证（正式 v2）','',
        '完成 Cora、CiteSeer、PubMed × GIN、GCN、GAT × 种子 41–43，共 27 组。新训练 21 个无补偿基线，各 200 轮；复用 6 个旧 GIN 基线；每组门控训练 100 轮。全部模型计算使用 GPU。','',
        '只训练门控相对无补偿的测试准确率变化，按 Cora、CiteSeer、PubMed 顺序：'+'；'.join(findings)+'。','',
        '本轮结果依赖模型：GCN 平均有所提高，GAT 平均略降；正向变化的配对 95% 区间仍包含零，尚不能认为获得稳定收益。Cora 的 GCN 加位宽对照均值高于门控，因此门控也未显示一致的消息预算优势。局部误差下降与分类收益仍不一致。','',
        '所有 27 组验证选模和检查点锁定后，统一评估四个预定条件，共 108 个测试条件；各条件在本版本记录一次。','',
        '## 修订与评估边界','',
        'v1 全部结果保留于 ../residual_transfer_20260914_v1。位宽核对发现部分 GCN 的 signed bit 舍入到 0，原量化函数此时 qmax=-0.5 却仍产生非零输出，因此不能当作有效零位编码。v2 对所有新训练基线的可训练 signed bit 参数统一投影到 [1,16]，按相同矩阵完整重跑。修订基于编码合法性，不根据准确率选择配置或版本。','',
        'v1 测试结果已查看，公共基准及复用 GIN 也有既有实验历史。本报告是探索性跨模型验证，不能描述为从未使用过的独立测试集。只有三个种子；配对 95% t 区间未经多重比较校正。','',
        '## 测试准确率','',
        '表内为三种子均值 ± 样本标准差（%）；门控模型按最低验证交叉熵选模，包含零门控的 epoch 0。','',
        '| 数据集 | 模型 | 无补偿 | 全量补偿 | 只训练门控 | 同消息预算加位宽 | 门控−基线（百分点） |',
        '|---|---|---:|---:|---:|---:|---:|']
    for d in DATASETS:
        for m in MODELS:
            g=group(d,m);delta=next(c['test_accuracy_delta']['mean'] for c in selected if c['dataset']==d and c['model']==m)
            text.append(f'| {d} | {m} | '+' | '.join(meanstd(g,v) for v in MODES)+f' | {delta:+.3f} |')
    text.extend(['','## 配对比较','','| 数据集 | 模型 | 当前 / 对照 | 测试准确率差 [95% 区间] | 胜/平/负 |','|---|---|---|---:|---:|'])
    for c in comparisons:
        a=c['test_accuracy_delta'];lo,hi=a['ci95']
        text.append(f"| {c['dataset']} | {c['model']} | {LABELS[c['method']]} / {LABELS[c['reference']]} | {a['mean']:+.3f} [{lo:+.3f}, {hi:+.3f}] | {a['wins']}/{a['ties']}/{a['losses']} |")
    text.extend(['','零方差区间只表示这三个配对结果完全一致，不能据此排除其他种子的变化。门控优于全量补偿可能只是关闭了有害通道；必须同时看无补偿和加位宽对照。','',
        '## 门控及选模','','| 数据集 | 模型 | 种子 | 基线最佳轮次 | 门控最佳轮次 | 所选门控 | 最佳验证准确率 % | 最后验证准确率 % |','|---|---|---:|---:|---:|---|---:|---:|'])
    for r in records:
        text.append(f"| {r['dataset']} | {r['architecture']} | {r['seed']} | {r['base']['epoch']} | {r['best_gate_epoch']} | "+', '.join(f'{v:.4f}' for v in r['gates'])+f" | {100*r['validation']['gate']['accuracy']:.3f} | {100*r['last_validation']['accuracy']:.3f} |")
    text.extend(['',f"27 组中 {verification['zero_gate_selections']} 组选回全零门控；这些情况是恢复原模型。{verification['last_epoch_gate_selections']} 组在第 100 轮选中最佳结果，不能保证门控已经充分收敛。",'',
        'Cora、CiteSeer 的 GIN 保留历史两层实现及旧检查点。这些基线弱于本轮相应 GCN/GAT 基线；对其改善不代表超过强 GNN 基线，也不宜用不同模型的绝对准确率解释补偿机制。','',
        '## 局部补偿误差','','各项先在每个模型自身当前层输入处计算，再对种子取均值。GAT 固定当前注意力系数，仅绕过值消息量化；不衡量注意力误差或完整 FP32 模型差异。','',
        '| 数据集 | 模型 | 条件 | 各量化层平方误差下降 % |','|---|---|---|---|'])
    for d in DATASETS:
        for m in MODELS:
            g=group(d,m)
            for mode in ('unit','gate'):
                values=[]
                for layer in range(len(g[0]['validation'][mode]['local_diagnostics'])):
                    a=[r['validation'][mode]['local_diagnostics'][layer]['mse_reduction'] for r in g]
                    values.append('未定义' if all(x is None for x in a) else f'{100*np.mean([x for x in a if x is not None]):.3f}')
                text.append(f'| {d} | {m} | {LABELS[mode]} | '+', '.join(values)+' |')
    text.extend(['','GIN 第一层未量化消息，表中只有第二层对应的一个量化块。GAT 的基为每头 rank 2，第一层总编码秩为 16；门控仍每层一个标量。补偿前后误差及带门控正交分解全部核对，关闭诊断时不计算完整宽度参考聚合。','',
        '## 活跃状态与消息预算','','| 数据集 | 模型 | 门控数 | 门控方案新增状态字节 | 加位宽配置字节 | 门控方案各层额外位/节点 |','|---|---|---:|---:|---:|---|'])
    for d in DATASETS:
        for m in MODELS:
            r=group(d,m)[0];gate=r['test']['gate']['bit_audit'];high=r['test']['higher']['bit_audit']
            text.append(f"| {d} | {m} | {r['gate_parameters']} | {gate['active_extra_state_bytes']} | {high['active_extra_state_bytes']} | "+', '.join(str(l['side_bits_per_node']) for l in gate['layers'])+' |')
    text.extend(['',
        '新增状态包括固定基、尺度和门控；加位宽配置保存 INT64 的逐通道增量。通用检查点额外保留所有对照的校准元数据，活跃状态字节不等于文件大小。INT8 侧码先按边读取，再在接收侧解码；GIN 无权邻居求和用 INT32，GCN/GAT 带权聚合使用浮点。这里不宣称端到端整数推理或测得通信加速。','',
        '每层加位宽增量总和严格匹配侧码预算，窄输出层允许每通道增加多位。GAT 加位宽对照使用更高精度值计算注意力，补偿组使用基础量化值计算注意力；因此加位宽可能同时改善注意力。GAT 第一层侧码为 128 位/节点，开销不能按普通单头 rank 2 的 16 位计算。','',
        '各条件实际位宽直方图及逻辑量化张量流量保存在 result.json 的 test.*.bit_audit。它们不是峰值显存、网络带宽或实际压缩文件大小。即使门控为零，参考实现仍计算侧码。','',
        '## 实现与验收','',
        '12 项正式 GPU 单元测试通过。核对 27 组冻结骨干、校准基和尺度，逐张量验证最佳检查点与最小验证损失选择；训练节点校准、合法 signed 位宽、精确预算匹配均通过。源检查点和训练/分析源码哈希保持不变。','',
        '另在 GPU 比较全部 108 个条件的完整图输出：先读 INT8 边侧码再解码，与先解码节点码的数学参考逐值一致；验证指标与训练记录精确复现，不重新计算测试指标。','',
        '训练配置和方法边界见 METHOD.md；逐种子结果见 per_seed.csv；机器可读汇总见 analysis.json；数值验收见 verification.json；复现及恢复命令见 RUN_COMMANDS.md。','',
        '## 图表','',f'![测试对照]({root}/test_comparison.png)','',f'![配对差值]({root}/paired_differences.png)','',f'![局部误差谱]({root}/error_spectra.png)',''])
    for d in DATASETS:text.extend([f'![{d} 门控]({root}/{d}_gates.png)',''])
    old=root.parent/'residual_transfer_20260914_v1'
    text.extend(['## 首版保留结果','',
        '下表保留修订前均值，便于核查版本变化；不从两版中按准确率挑选结果。首版部分 GCN 含无效零位参数，正式结论以修正后的 v2 为准。','',
        '| 数据集 | 模型 | v1 无补偿 % | v1 门控 % | v1 门控−基线 |','|---|---|---:|---:|---:|'])
    for d in DATASETS:
        for m in MODELS:
            previous=[json.loads((old/d/m/f'seed_{s}/result.json').read_text()) for s in SEEDS]
            before=np.mean([r['test']['none']['accuracy'] for r in previous])*100
            after=np.mean([r['test']['gate']['accuracy'] for r in previous])*100
            text.append(f'| {d} | {m} | {before:.3f} | {after:.3f} | {after-before:+.3f} |')
    (root/'REPORT.md').write_text('\n'.join(text)+'\n')
    commands=f'''# 复现与恢复

全部模型计算使用 GPU。恢复既有队列：

```sh
cd {root}/source_snapshot
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.residual_transfer.suite --root {root} --gpus 0 1 2 3
```

复现单组训练到新目录（200 轮基线、100 轮门控，不释放测试）：

```sh
cd {root}/source_snapshot
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.residual_transfer.run --root /home/cyliu/code/A-2Q/result/lorap_study/residual_transfer_reproduction --dataset PubMed --model GCN --seed 41 --phase fit
```

完整复现实验矩阵：在第一条 suite 命令中将 --root 改为新的空输出目录。所有组完成后会统一锁定选模并评估四个测试条件。不要对原目录删除测试标记来重复测试。

仅读取现有结果生成报告：

```sh
cd {root}/analysis_snapshot
PYTHONDONTWRITEBYTECODE=1 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.residual_transfer.report --root {root}
```

GPU 执行路径等价审计（只核对完整输出和验证指标）：

```sh
cd {root}/analysis_snapshot
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=0 /home/cyliu/anaconda3/envs/ProG/bin/python -B -m runs.lorap_study.residual_transfer.audit --root {root}
```
'''
    (root/'RUN_COMMANDS.md').write_text(commands)
    print(json.dumps(dict(verification=verification,gate_vs_baseline=selected),ensure_ascii=False,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    analyze(parser.parse_args().root)
