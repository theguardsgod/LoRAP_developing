"""Verify all final records and report paired sample SD; no test inference."""
import csv
from pathlib import Path
import statistics

import torch

from . import portable as p


def stats(values):
    return dict(mean=statistics.mean(values),sample_sd=statistics.stdev(values),n=len(values))


def report(root):
    root=Path(root)
    study=p.read(root/'study.json');completion=p.read(root/'completion_audit.json')
    assert completion['passed'] and completion['fits']==completion['tests']==6
    assert study['seeds']==[41,42,43] and study['cells']==['QAT_MNIST_GAT']
    assert not study['smoke'] and study['subset_limit'] is None
    gate=p.read(root/'STABILITY_GATE.json')
    assert gate['passed'] and gate['study_hash']==p.config_hash(study)
    for relative,expected in p.read(root/'source_manifest.json').items():
        assert p.sha(root/'source_snapshot'/relative)==expected
    manifest=p.test_manifest(root,study)
    assert manifest==p.read(root/'TEST_LOCK.json')
    rows=[]
    for job in study['jobs']:
        folder=root/job['directory']
        assert not (folder/'failure.json').exists()
        history=p.read(folder/'history.json');done=p.read(folder/'done.json');test=p.read(folder/'test.json')
        assert len(history)==job['epochs']==200
        assert [h['epoch'] for h in history]==list(range(1,201))
        best=min(history,key=lambda h:(h['validation']['loss'],h['epoch']))
        assert done['selected_epoch']==best['epoch']
        assert abs(done['validation']['loss']-best['validation']['loss'])<1e-5
        assert test['completed_test_evaluations']==1 and test['seed']==job['seed']
        assert test['checkpoint_file_sha256']==p.sha(folder/'best.pt')==done['checkpoint_file_sha256']
        assert test['test_lock_sha256']==manifest['sha256']
        split=p.read(folder/'split.json')
        assert split['train_graphs']==55000 and split['val_graphs']==5000 and test['examples']==10000
        assert all(h['train']['examples']==55000 and h['validation']['examples']==5000 for h in history)
        raw=max(h['train']['gradient_norm_max'] for h in history)
        clipped=max(h['train']['gradient_norm_after_clip_max'] for h in history)
        assert raw<=1e6 and clipped<=1.00001
        last=torch.load(folder/'last.pt',map_location='cpu',weights_only=False)
        assert last['epoch']==200
        def finite_tree(value):
            if torch.is_tensor(value): assert torch.isfinite(value).all().item()
            elif isinstance(value,dict):
                for v in value.values():finite_tree(v)
        finite_tree(last['model']);finite_tree(last['optimizer'])
        rows.append(dict(seed=job['seed'],variant=job['variant'],test_accuracy=test['metric'],
            validation_loss=done['validation']['loss'],selected_epoch=done['selected_epoch'],
            raw_gradient_peak=raw,clipped_gradient_peak=clipped,
            clipping_fraction=sum(h['train']['gradient_clipped_batches'] for h in history)/
                              sum(h['train']['gradient_batches'] for h in history),
            training_seconds=done['training_seconds'],peak_cuda_memory_bytes=done['peak_cuda_memory_bytes'],
            adapter_parameters=done['adapter_parameters'],device=done['device']))
    paired=[]
    for seed in study['seeds']:
        values={r['variant']:r['test_accuracy'] for r in rows if r['seed']==seed}
        paired.append(dict(seed=seed,none=values['none'],graphlora=values['graphlora'],
                           difference_pp=values['graphlora']-values['none']))
    summary={v:stats([r['test_accuracy'] for r in rows if r['variant']==v]) for v in p.VARIANTS}
    summary['paired_difference_pp']=stats([r['difference_pp'] for r in paired])
    p.write_json(root/'paired_summary.json',summary)
    for name,records in [('per_seed.csv',rows),('paired_differences.csv',paired)]:
        with (root/name).open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    lines=['# MNIST–GAT：稳定化配方配对结果','','GraphLoRA-adapted v2，joint QAT；均值 ± 样本标准差，准确率单位 %。',
        '新初始化；完整官方划分；种子 41–43；均训练 200 轮。两种方法均用学习率 0.001、全局梯度裁剪 1.0，原始梯度超过 1e6 仍拒绝。',
        '前 30 轮仅作数值筛查，通过后从本次新实验断点继续，不复用修复前 Adam。全部配对训练完成后各测试一次。','',
        '| 方法 | 准确率 / 配对差值（mean ± sample SD） |','|---|---:|']
    for v,s in summary.items():lines.append(f"| {v} | {s['mean']:.3f} ± {s['sample_sd']:.3f} |")
    lines+=['','| 方法 | 最佳验证 CE（mean ± SD） | 原始梯度峰值 | 裁剪后峰值 | 平均裁剪比例 |','|---|---:|---:|---:|---:|']
    for v in p.VARIANTS:
        group=[r for r in rows if r['variant']==v];val=stats([r['validation_loss'] for r in group])
        lines.append(f"| {v} | {val['mean']:.6f} ± {val['sample_sd']:.6f} | {max(r['raw_gradient_peak'] for r in group):.6g} | {max(r['clipped_gradient_peak'] for r in group):.6g} | {statistics.mean(r['clipping_fraction'] for r in group):.2%} |")
    lines+=['','完整流程审计通过；最终模型与 Adam 状态为有限值。数值稳定不代表准确率提升或统计显著性。',
        '此配方改变了训练协议，不能替代指定提交 70ae58df 的原始失败结果，也不与旧实验历史 None 混作配对比较。',
        '源码基线、改动补丁哈希、环境与实际 GPU 见 provenance.json；逐种子资源和梯度见 per_seed.csv。','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    p.write_json(root/'post_completion_audit.json',dict(passed=True,fits=6,tests=6,paired_initializations_and_splits=3,
        full_official_splits=True,all_200_epochs=True,earliest_minimum_validation_loss=True,
        finite_final_adam=True,raw_and_clipped_gradient_limits=True,single_test_checkpoint_hashes=True,
        sample_sd=True,test_inferences_performed_by_report=0))
