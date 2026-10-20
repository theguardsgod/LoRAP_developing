"""Generate a complete Chinese report only after all locked tests finish."""
import csv,json,math,statistics
from pathlib import Path
from .registry import CELLS,METHODS,FRAMEWORKS
from .common import write_json


LABELS=dict(none='本轮无提示',lorap='原 LoRAP',gpf='GPF-plus',gpf_lorap='GPF-plus + LoRAP',
            conditioned='零初始化条件 LoRAP',spectral='误差谱基 LoRAP',task='量化预测任务版',
            error='量化预测误差监督版',mlp='等容量 MLP',weight_lora='权重 LoRA',gate_only='预测后仅训练门控')


def backbone_search_notes(root):
    """Disclose validation-driven backbone changes separately from adaptation."""
    lines=[]
    for name in CELLS:
        path=root/name/'baseline_search/selection.json'
        if not path.exists():continue
        result=json.loads(path.read_text())
        if not lines:
            lines=['','## 冻结前的骨干调参','',
                '两组 A²Q GIN 因验证准确率与最低 CE 选模差距超过 10 个百分点，按事先记录的补充协议各增加 5 个骨干候选。只按种子 41 的最低验证 CE 选择；架构、量化预算与量化器学习率不变，种子 42、43 固定采用所选配方。所有冻结方法共享该骨干，原始起点和全部尝试均保留。以下均为验证数据，不能与论文测试准确率直接比较。','',
                '| 配置 | 候选 | 主学习率 | 权重衰减 | 验证 CE | 验证准确率 (%) | 选中 |',
                '|---|---:|---:|---:|---:|---:|---|']
        for row in result['candidates']:
            hp=row['hyperparameters'];val=row['validation']
            lines.append(f"| {name} | {row['candidate']} | {hp['lr']} | {hp['weight_decay']} | {val['loss']:.6f} | {val['metric']:.2f} | {'是' if row['candidate']==result['selected']['candidate'] else ''} |")
    return lines


def generate(root):
    root=Path(root);lock=json.loads((root/'TEST_LOCK.json').read_text());records={};diagnostics={}
    for case in lock['cases']:
        path=root/case['directory']
        if not (path/'final_complete.json').exists():raise RuntimeError('A complete report requires all 1485 final evaluations')
        test=json.loads((path/'test.json').read_text())
        records[(case['cell'],case['method'],case['seed'])]=test
        if (path/'diagnostics.json').exists():
            diagnostics[(case['cell'],case['method'],case['seed'])]=json.loads((path/'diagnostics.json').read_text())
    rows=[];invariants=[]
    for name,c in CELLS.items():
        none=[records[name,'none',s]['metric'] for s in c['seeds']]
        for method in ('none',*METHODS):
            values=[records[name,method,s]['metric'] for s in c['seeds']]
            mean=statistics.mean(values);sd=statistics.stdev(values)
            direction=-1 if c['regression'] else 1
            deltas=[direction*(a-b) for a,b in zip(values,none)]
            gain=statistics.mean(deltas);radius=4.302652729*statistics.stdev(deltas)/math.sqrt(3)
            rows.append(dict(cell=name,dataset=c['dataset'],architecture=c['architecture'],framework=c['framework'],method=method,
                mean=mean,sd=sd,seed_41=values[0],seed_42=values[1],seed_43=values[2],paper_none=c['paper_none'],
                oriented_paper_gain=direction*(mean-c['paper_none']),paired_gain=gain,paired_ci_low=gain-radius,paired_ci_high=gain+radius,
                paper_exceeded=direction*(mean-c['paper_none'])>0,paired_improved=gain>0,
                none_already_exceeds_paper=direction*(statistics.mean(none)-c['paper_none'])>0))
        for seed in c['seeds']:
            reference=None
            for method in METHODS:
                data=diagnostics.get((name,method,seed),{}).get('fixed_input',{})
                if not data.get('applicable'):continue
                before=[layer['mse_before'] for layer in data['layers']]
                if reference is None:reference=before
                passed=len(before)==len(reference) and all(abs(a-b)<=1e-12*max(1,abs(a),abs(b)) for a,b in zip(before,reference))
                invariants.append(dict(cell=name,seed=seed,method=method,same_reference=passed))
    if not all(r['same_reference'] for r in invariants):raise AssertionError('Fixed-input reference mismatch')
    write_json(root/'summary.json',rows);write_json(root/'fixed_input_audit.json',invariants)
    with (root/'summary.csv').open('w') as file:
        writer=csv.DictWriter(file,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    figures=plots(root,rows,diagnostics)
    lines=['# 表 I 全部 45 组：冻结骨干对照结果','',
       '五个数据集 × 三种架构 × 三种量化框架；种子 41、42、43。所有测试来自验证选模后锁定的检查点。分类为准确率（%），ZINC 为 MAE；下列差值统一为正数表示改进。',
       '', '本轮为探索性三种子冻结适配，论文表 I 是联合训练，且节点/图实验重复次数为 100/10。不能把本轮标准差或正均值差当作论文级显著收益。',
       '', '| 方法 | 超过论文 None /45 | 提升本轮 None /45 | 两者同时满足 /45 |', '|---|---:|---:|---:|']
    for method in METHODS:
        subset=[r for r in rows if r['method']==method]
        lines.append(f"| {LABELS[method]} | {sum(r['paper_exceeded'] for r in subset)} | {sum(r['paired_improved'] for r in subset)} | {sum(r['paper_exceeded'] and r['paired_improved'] for r in subset)} |")
    lines+=backbone_search_notes(root)
    lines+=['','“超过论文 None”与“适配器带来收益”分别统计；若本轮无提示起点本来就超过论文参考，不能把全部差距归因于冻结方法。',
            '',f'![两种基线的胜出分布]({figures[0]})','','## 全部结果','']
    for name,c in CELLS.items():
        lines += [f"### {name}",'',f"论文 None：{c['paper_none']} ± {c['paper_none_sd']}；论文 GPF-LoRAP：{c['paper_gpf_lorap']}。",'',
                  '| 方法 | 本轮均值 ± 标准差 | 相对论文 None | 相对本轮 None | 配对差 95% CI |', '|---|---:|---:|---:|---|']
        precision=4 if c['regression'] else 2
        for r in [row for row in rows if row['cell']==name]:
            fmt=lambda x:f'{x:.{precision}f}'
            lines.append(f"| {LABELS[r['method']]} | {fmt(r['mean'])} ± {fmt(r['sd'])} | {fmt(r['oriented_paper_gain'])} | {fmt(r['paired_gain'])} | [{fmt(r['paired_ci_low'])}, {fmt(r['paired_ci_high'])}] |")
        spectrum=root/'figures'/f'{name}_spectrum.png'
        if spectrum.exists():lines+=['',f'![固定输入的局部误差谱，种子 41]({spectrum})','']
    lines+=['','## 误差补偿的解释边界','',
       '误差诊断固定当前层输入、权重、注意力，只绕过聚合前消息量化；DQ/QAT 的后续聚合量化保持开启。固定输入对照不传播补偿，确保各预测器看到相同骨干输入。误差降低表示局部量化误差得到补偿，不等价于完整 FP32 模型被恢复，也不保证测试性能提升。',
       '', '原始 GPF、权重 LoRA 的修改位置与聚合预测器不同，因此固定聚合输入补偿指标标为不适用，不能用零值代替。MLP 与 task 的有效参数量差异小于 5%；真实数值、归一化缓冲区和位宽分布见各运行 initialization.json、quantization.json。',
       '', '推理导出物不包含标签或观测到的训练误差向量。所有导出模型通过验证数据的逐值重放和无标签输入检查；正常推理不执行教师分支。',
       '', '## 可复现文件','',
       f'- [锁定的测试清单]({root/"TEST_LOCK.json"})',f'- [全部逐种子汇总]({root/"summary.csv"})',
       f'- [相同参考输入审计]({root/"fixed_input_audit.json"})',
       f'- [协议]({Path(__file__).resolve().parent/"PROTOCOL.md"})',
       f'- [后端差异说明]({Path(__file__).resolve().parent/"BACKEND_NOTES.md"})',
       f'- [运行命令]({Path(__file__).resolve().parent/"README.md"})','']
    (root/'FINAL_REPORT.md').write_text('\n'.join(lines))


def plots(root,rows,diagnostics):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    folder=root/'figures';folder.mkdir(exist_ok=True)
    names=list(CELLS);grid=np.zeros((len(names),len(METHODS)))
    for r in rows:
        if r['method']=='none':continue
        grid[names.index(r['cell']),METHODS.index(r['method'])]=int(r['paper_exceeded'])+2*int(r['paired_improved'])
    fig,ax=plt.subplots(figsize=(11,15));cm=ListedColormap(['#e5e7eb','#f2c14e','#73b5d5','#309568'])
    im=ax.imshow(grid,cmap=cm,norm=BoundaryNorm([-.5,.5,1.5,2.5,3.5],4),aspect='auto')
    ax.set_xticks(range(len(METHODS)),METHODS,rotation=45,ha='right');ax.set_yticks(range(len(names)),names,fontsize=8)
    colorbar=fig.colorbar(im,ax=ax,ticks=[0,1,2,3],fraction=.025,pad=.02)
    colorbar.ax.set_yticklabels(['Neither','Paper only','Paired only','Both'])
    ax.set_title('Mean test comparisons; 3 paired seeds; positive mean is not significance')
    fig.tight_layout();overview=folder/'wins.png';fig.savefig(overview,dpi=180);plt.close(fig)
    for name,c in CELLS.items():
        available={m:diagnostics.get((name,m,41),{}).get('fixed_input',{}) for m in ('task','error','mlp')}
        if not all(d.get('applicable') for d in available.values()):continue
        n=len(available['task']['layers']);fig,axes=plt.subplots(1,n,figsize=(4*n,3.3),squeeze=False)
        for i,ax in enumerate(axes[0]):
            before=np.array(available['task']['layers'][i]['before_singular_values'])**2
            denom=max(before.sum(),1e-30);ax.plot(np.arange(1,len(before)+1),before/denom,label='Uncorrected',color='black')
            for method in ('task','error','mlp'):
                spectrum=np.array(available[method]['layers'][i]['after_singular_values'])**2
                ax.plot(np.arange(1,len(spectrum)+1),spectrum/denom,label=method)
            ax.set_yscale('symlog',linthresh=1e-5);ax.set_xlabel('Singular-value index');ax.set_ylabel('Energy / uncorrected total')
            ax.set_title(f'Aggregation site {i+1}');ax.legend(fontsize=8)
        fig.suptitle(f'{name}, seed 41, fixed no-prompt layer inputs')
        fig.tight_layout();fig.savefig(folder/f'{name}_spectrum.png',dpi=170);plt.close(fig)
    return [str(overview)]


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);generate(Path(p.parse_args().root))
