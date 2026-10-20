"""Validation-only checkpoint/calibration sensitivity; never changes selections."""
import argparse,json
from pathlib import Path
import torch
from torch.nn import functional as F
from .registry import CELLS
from .data import Bundle
from .backends import build,load_state,predict
from .common import write_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--device',default='cuda:2')
    a=p.parse_args();root=Path(a.root);torch.set_num_threads(4)
    rows=[];calibration=[]
    for name,c in CELLS.items():
        path=root/name/'seed_41/baseline'
        if not c['node'] or not (path/'done.json').exists():continue
        done=json.loads((path/'done.json').read_text());history=json.loads((path/'history.json').read_text())
        highest=max(history,key=lambda r:(r['val']['metric'],-r['val']['loss'],-r['epoch']))
        rows.append(dict(cell=name,selected_epoch=done['selected_epoch'],selected_val_accuracy=done['validation']['metric'],
                         selected_val_loss=done['validation']['loss'],highest_val_accuracy=highest['val']['metric'],
                         highest_accuracy_epoch=highest['epoch'],validation_gap=highest['val']['metric']-done['validation']['metric']))
        if name not in ('A2Q_Cora_GIN','A2Q_CiteSeer_GIN'):continue
        b=Bundle(c);model=build(c,b,41).to(a.device).eval();batch=next(b.batches('val',41,0,a.device))
        for checkpoint_name in ('best','last'):
            state=torch.load(path/(checkpoint_name+'.pt'),map_location='cpu',weights_only=False)
            load_state(model,state['model']);out,_=predict(model,c,batch)
            logits,y=out[batch.val_mask],batch.y[batch.val_mask]
            for temperature in (.5,1.,2.,4.,8.,16.):
                calibration.append(dict(cell=name,checkpoint=checkpoint_name,epoch=state['epoch'],temperature=temperature,
                    validation_loss=float(F.cross_entropy(logits/temperature,y)),
                    validation_accuracy=float((logits.argmax(1)==y).float().mean()*100)))
    directory=root/'baseline_validation_audit';directory.mkdir(exist_ok=True)
    write_json(directory/'selection_sensitivity.json',rows);write_json(directory/'temperature_sensitivity.json',calibration)
    lines=['# 节点骨干的验证选模诊断（种子 41）','',
           '这里只读取训练历史和验证标签；没有测试集指标，也没有改变任何正式 checkpoint 或超参数。最高验证准确率仅作曲线诊断。', '',
           '| 配置 | 最低 CE 所选轮次 | 所选验证准确率 | 曲线最高验证准确率 | 其轮次 | 差距（百分点） |',
           '|---|---:|---:|---:|---:|---:|']
    for r in rows:
        lines.append(f"| {r['cell']} | {r['selected_epoch']} | {r['selected_val_accuracy']:.2f} | {r['highest_val_accuracy']:.2f} | {r['highest_accuracy_epoch']} | {r['validation_gap']:.2f} |")
    lines+=['','## GIN 置信度敏感性','',
            '对已保存的最低 CE 和末轮权重仅在验证时缩放 logits，温度不会改变 argmax 或准确率。这是损失对置信度的敏感性检查，**未用于选择正式基线，也不是新增方法的结果**。', '',
            '| 配置 | checkpoint | T=1 验证 CE | 网格内最低 CE | 对应 T | 验证准确率（不变） |', '|---|---|---:|---:|---:|---:|']
    for name in ('A2Q_Cora_GIN','A2Q_CiteSeer_GIN'):
        for which in ('best','last'):
            options=[r for r in calibration if r['cell']==name and r['checkpoint']==which]
            if not options:continue
            original=next(r for r in options if r['temperature']==1);best=min(options,key=lambda r:r['validation_loss'])
            lines.append(f"| {name} | {which} | {original['validation_loss']:.4f} | {best['validation_loss']:.4f} | {best['temperature']} | {best['validation_accuracy']:.2f} |")
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,6))
    for column,name in enumerate(('A2Q_Cora_GIN','A2Q_CiteSeer_GIN')):
        history=json.loads((root/name/'seed_41/baseline/history.json').read_text());epochs=[r['epoch'] for r in history]
        chosen=next(r for r in rows if r['cell']==name)['selected_epoch']
        for axis,field,label in ((axes[0,column],'loss','Validation cross-entropy'),(axes[1,column],'metric','Validation accuracy (%)')):
            axis.plot(epochs,[r['val'][field] for r in history]);axis.axvline(chosen,color='#bf5b17',linestyle='--',label='Selected by minimum CE')
            axis.set_ylabel(label);axis.set_xlabel('Epoch');axis.legend(fontsize=8)
        axes[0,column].set_title(name)
    fig.tight_layout();figure=directory/'gin_validation_selection.png';fig.savefig(figure,dpi=180);plt.close(fig)
    lines+=['',f'![GIN 验证损失与准确率]({figure.resolve()})','']
    (directory/'REPORT.md').write_text('\n'.join(lines));print('\n'.join(lines[:25]))


if __name__=='__main__':main()
