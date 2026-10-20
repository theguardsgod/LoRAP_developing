"""Append matched no-prompt controls to an immutable completed Table V LoRA study."""
import argparse
import csv
import fcntl
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import time

import torch
from torch_geometric.data import Data

from .full_run import sha_config
from .lora import baseline_counts, baseline_optimizer, build_baseline, common_backbone_digest
from .lora_run import fit, release_tests, require_device, study_data, test_once
from .run import write_json

MODES = ('fixed4', 'a2q')


def read(path):
    return json.loads(Path(path).read_text())


def inherited_config(reference, smoke=False):
    if reference['variants'] != ['lora'] or reference['subset_sizes'] is not None:
        raise ValueError('Reference must be the completed full-data LoRA protocol')
    config = dict(reference, variants=['no_prompt'])
    if smoke:
        config.update(seeds=reference['seeds'][:1], epochs=2,
                      subset_sizes=dict(train=128, val=64, test=64))
    return config


def compare_pair(reference, control):
    for key in ('dataset', 'seed', 'epochs', 'common_backbone_sha256'):
        if reference[key] != control[key]:
            raise AssertionError(f'Unmatched {key}')
    assert control['prompt_parameters'] == control['lora_parameters'] == 0
    assert reference['total_parameters'] - control['total_parameters'] == reference['lora_parameters']
    assert reference['trainable_parameters'] - control['trainable_parameters'] == reference['lora_parameters']
    assert reference['test_evaluations'] == control['test_evaluations'] == 1
    return dict(seed=control['seed'], no_prompt_accuracy_percent=100*control['test']['accuracy'],
                lora_accuracy_percent=100*reference['test']['accuracy'],
                lora_minus_no_prompt_pp=100*(reference['test']['accuracy']-control['test']['accuracy']),
                lora_minus_no_prompt_CE=reference['test']['loss']-control['test']['loss'])


def stats(values):
    return dict(mean=statistics.mean(values), sample_std=statistics.stdev(values) if len(values)>1 else None) if values else None


def display(value):
    if value is None:
        return '待测'
    return f"{value['mean']:.2f} ± {value['sample_std']:.2f}" if value['sample_std'] is not None else f"{value['mean']:.2f} (n=1)"


def summarize(root):
    suite = read(root/'suite.json')
    pairs, rows, progress, resources = [], [], [], []
    fit_total = test_total = 0
    for mode in MODES:
        results, references = [], []
        for seed in suite['seeds']:
            directory = root/mode/'MNIST/no_prompt'/f'seed_{seed}'
            result = None
            reference = read(root/'reference'/mode/'MNIST/lora'/f'seed_{seed}'/'result.json')
            references.append(reference)
            if (directory/'progress.json').exists():
                progress.append(dict(mode=mode, seed=seed, **read(directory/'progress.json')))
            if (directory/'result.json').exists():
                result = read(directory/'result.json')
                assert result['common_backbone_sha256'] == reference['common_backbone_sha256']
                assert result['lora_parameters'] == result['prompt_parameters'] == 0
                results.append(result)
                if result['test_evaluations'] and not suite['smoke']:
                    pairs.append(dict(mode=mode, **compare_pair(reference, result)))
            for variant, record in [('lora', reference)] + ([('no_prompt', result)] if result is not None else []):
                resources.append(dict(mode=mode, variant=variant, seed=seed,
                    total_parameters=record['total_parameters'], trainable_parameters=record['trainable_parameters'],
                    prompt_parameters=record['prompt_parameters'], lora_parameters=record['lora_parameters'],
                    extra_fp32_bytes=record['prompt_fp32_bytes']+record['lora_fp32_bytes'],
                    activation_effective_bits=record['mean_activation_effective_bits'],
                    training_seconds=record['training_seconds'], peak_cuda_memory_bytes=record['peak_cuda_memory_bytes'],
                    device=record['device']))
        tested = [r for r in results if r['test_evaluations']==1]
        fit_total += len(results)
        test_total += len(tested)
        ready = len(tested)==len(suite['seeds']) and not suite['smoke']
        paired = [p for p in pairs if p['mode']==mode]
        rows.append(dict(mode=mode, fits=len(results), tests=len(tested), ready=ready,
            no_prompt_accuracy=stats([100*r['test']['accuracy'] for r in tested]) if ready else None,
            lora_accuracy=stats([100*r['test']['accuracy'] for r in references]) if not suite['smoke'] else None,
            paired_difference_pp=stats([p['lora_minus_no_prompt_pp'] for p in paired]) if ready else None,
            no_prompt_activation_bits=stats([r['mean_activation_effective_bits'] for r in results]) if ready else None,
            lora_activation_bits=stats([r['mean_activation_effective_bits'] for r in references]),
            lora_wins=sum(p['lora_minus_no_prompt_pp']>0 for p in paired) if ready else None,
            lora_losses=sum(p['lora_minus_no_prompt_pp']<0 for p in paired) if ready else None))
    expected = 2*len(suite['seeds'])
    complete = fit_total==expected and (suite['smoke'] or test_total==expected)
    write_json(root/'status.json', dict(complete=complete, smoke=suite['smoke'], fits_completed=fit_total,
        fits_total=expected, tests_completed=test_total, tests_expected=0 if suite['smoke'] else expected, rows=rows, progress=progress))
    write_json(root/'comparison.json', rows)
    for name, items in [('paired_differences.csv', pairs), ('resources.csv', resources)]:
        if items:
            with (root/name).open('w', newline='') as stream:
                writer=csv.DictWriter(stream, fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    report=['# 无 LoRA 对照：与既有 Table V LoRA 实验配对', '',
        f"新增训练 {fit_total}/{expected}；新增测试 {test_total}/{0 if suite['smoke'] else expected}；{'已完成' if complete else '进行中'}。", '',
        f"参考实验：{suite['reference_root']}。完整 MNIST 55,000/5,000/10,000 图，200 轮，种子 41–50，batch size 128。", '',
        '每种量化协议直接继承原配置，正式实验仅将 variants 从 lora 改为 no_prompt。骨干与量化参数初始化逐种子匹配，'
        '无提示且无 LoRA，额外参数开销为 0。训练数据顺序、优化器基础分组、调度器、损失和选模规则沿用原源码。', '',
        '| 协议 | 无 LoRA 测试准确率 % | LoRA 测试准确率 % | 配对差值（LoRA − 无 LoRA，百分点） | 新增已训练 / 已测试 |',
        '|---|---|---|---|---|']
    for r in rows:
        report.append(f"| {'QAT / INT4' if r['mode']=='fixed4' else 'A2Q'} | {display(r['no_prompt_accuracy'])} | {display(r['lora_accuracy'])} | {display(r['paired_difference_pp'])} | {r['fits']} / {r['tests']} |")
    report += ['', '准确率与配对差值均报告均值 ± 样本标准差，逐种子差值见 paired_differences.csv；正值表示 LoRA 更高。'
        '参数量、实际位宽、耗时及显存逐组记录在 resources.csv。', '',
        '两种量化协议的所有新增训练完成后才释放测试；按最低验证交叉熵选检查点，平局取较早轮次，每个最佳检查点仅测试一次。'
        '既有 LoRA 结果仅从归档 JSON 读取，不重新训练或测试。', '',
        '这是看到 LoRA 结果后补充的固定协议对照，未据此调参；相同种子和配置不保证 GPU scatter 运算逐位确定。'
        '本轮内部比较使用相同位宽定义，但与原论文截图 Average bits 的定义仍需另行核对。', '',
        '原训练源码从参考实验的 source_snapshot 复制并逐文件校验，只新增本次调度/汇总代码。'
        '来源哈希、参考结果、配置差异、初始化配对检查和完整划分均保存于本目录。恢复命令见 RUN_COMMANDS.md。']
    if suite['smoke']:
        report += ['', '本次为 128/64/64 图、2 轮、首个种子的冒烟检查；不运行测试，不用于论文性能比较。']
    (root/'REPORT.md').write_text('\n'.join(report)+'\n')


def prepare(root, reference, smoke, device):
    original = read(reference/'suite.json')
    assert read(reference/'status.json')['complete'] and not original['smoke']
    provenance = read(reference/'provenance.json')
    for package, version in provenance['versions'].items():
        assert importlib.metadata.version(package)==version, f'Dependency changed: {package}'
    for name, expected in provenance['source_sha256'].items():
        assert hashlib.sha256((reference/'source_snapshot'/name).read_bytes()).hexdigest()==expected, name
    root.mkdir(parents=True, exist_ok=False)
    shutil.copytree(reference/'source_snapshot', root/'source_snapshot')
    module_name = 'runs/lorap_study/table_v_control.py'
    shutil.copy2(__file__, root/'source_snapshot'/module_name)
    hashes = dict(provenance['source_sha256'])
    hashes[module_name] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    write_json(root/'provenance.json', dict(reference_root=str(reference), original_provenance=provenance,
        source_sha256=hashes, created_unix=time.time(), python=sys.executable))
    suite=dict(seeds=original['seeds'][:1] if smoke else original['seeds'], epochs=2 if smoke else original['epochs'],
        device=device, threads=original['threads'], smoke=smoke, reference_root=str(reference))
    write_json(root/'suite.json', suite)
    torch.set_num_threads(suite['threads'])
    checks, reference_hashes = [], {}
    for mode in MODES:
        original_config=read(reference/mode/'MNIST/config.json')
        config=inherited_config(original_config, smoke)
        write_json(root/mode/'MNIST/config.json', config)
        write_json(root/mode/'protocol_diff.json', {k:dict(reference=original_config[k],control=v)
            for k,v in config.items() if original_config[k]!=v})
        for suffix in ['MNIST/config.json', 'MNIST/split_manifest.json'] + [
            f'MNIST/lora/seed_{s}/{name}' for s in suite['seeds']
            for name in ('result.json','initialization.json','optimizer_groups.json')]:
            source=reference/mode/suffix
            destination=root/'reference'/mode/suffix
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
            reference_hashes[str(destination.relative_to(root))]=hashlib.sha256(destination.read_bytes()).hexdigest()
        for seed in suite['seeds']:
            ref_dir=root/'reference'/mode/'MNIST/lora'/f'seed_{seed}'
            # Construction uses only feature dimensions; data contents do not affect initialization.
            model=build_baseline(config,'no_prompt',seed,{'train':[Data(x=torch.zeros(1,1),pos=torch.zeros(1,2))]})
            digest=common_backbone_digest(model)
            assert digest==read(ref_dir/'initialization.json')['common_backbone_sha256']
            counts=baseline_counts(model)
            assert counts['lora_parameters']==counts['prompt_parameters']==0
            optimizer=baseline_optimizer(model,config)
            actual=[dict(name=g['name'],lr=g['lr'],weight_decay=g['weight_decay'],parameters=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups]
            expected=[g for g in read(ref_dir/'optimizer_groups.json') if g['name']!='lora']
            assert actual==expected
            checks.append(dict(mode=mode,seed=seed,common_backbone_sha256=digest,optimizer_matches=True,**counts))
    write_json(root/'reference_hashes.json',reference_hashes)
    write_json(root/'pairing_preflight.json',dict(passed=True,checks=checks))
    (root/'RUN_COMMANDS.md').write_text(f'{sys.executable} -B -m runs.lorap_study.table_v_control --resume --output-dir {root}\n')
    summarize(root)


def worker(root):
    suite=read(root/'suite.json')
    torch.set_num_threads(suite['threads'])
    device=require_device(suite['device'])
    configs={m:read(root/m/'MNIST/config.json') for m in MODES}
    data,manifest=study_data(configs[MODES[0]])
    with (root/'.worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for mode in MODES:
            if not suite['smoke']:
                assert manifest==read(root/'reference'/mode/'MNIST/split_manifest.json')
            write_json(root/mode/'MNIST/split_manifest.json',manifest)
        for seed in suite['seeds']:
            for mode in MODES:
                write_json(root/'active.json',dict(phase='fit',mode=mode,seed=seed))
                result=fit(root/mode,configs[mode],'no_prompt',seed,data,None,device)
                reference=read(root/'reference'/mode/'MNIST/lora'/f'seed_{seed}'/'result.json')
                assert result['common_backbone_sha256']==reference['common_backbone_sha256']
        if not suite['smoke']:
            for mode in MODES:
                release_tests(root/mode,configs[mode])
            write_json(root/'test_release.json',dict(modes=list(MODES),seeds=suite['seeds']))
            for seed in suite['seeds']:
                for mode in MODES:
                    write_json(root/'active.json',dict(phase='test',mode=mode,seed=seed))
                    test_once(root/mode,configs[mode],'no_prompt',seed,data,None,device)
        write_json(root/'active.json',dict(phase='complete'))


def coordinator(root):
    with (root/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for name, expected in read(root/'provenance.json')['source_sha256'].items():
            assert hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest()==expected
        for name, expected in read(root/'reference_hashes.json').items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==expected
        command=[sys.executable,'-B','-m','runs.lorap_study.table_v_control','--worker-root',str(root)]
        with (root/'worker.log').open('a') as stream:
            process=subprocess.Popen(command,cwd=root/'source_snapshot',stdout=stream,stderr=subprocess.STDOUT)
            write_json(root/'worker.json',dict(pid=process.pid,started_unix=time.time()))
            try:
                while process.poll() is None:
                    summarize(root)
                    time.sleep(10)
                summarize(root)
                if process.returncode:
                    raise RuntimeError(f'Control worker failed ({process.returncode}); inspect worker.log')
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-root',type=Path)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker-root',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--coordinator-root',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker_root:
        worker(args.worker_root);return
    if args.coordinator_root:
        coordinator(args.coordinator_root);return
    if args.output_dir is None:
        parser.error('--output-dir is required')
    root=args.output_dir.resolve()
    if not args.resume:
        if args.reference_root is None:
            parser.error('--reference-root is required for a new control study')
        require_device(args.device)
        prepare(root,args.reference_root.resolve(),args.smoke,args.device)
    elif not (root/'suite.json').exists():
        parser.error('No saved control study to resume')
    subprocess.run([sys.executable,'-B','-m','runs.lorap_study.table_v_control','--coordinator-root',str(root)],
                   cwd=root/'source_snapshot',check=True)


if __name__=='__main__':
    main()
