"""GLoRA joint-QAT: supernet training, validation evolution, compact one-shot test."""
import argparse
import fcntl
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

import torch
from torch_geometric.data import Data

from .data import loaders_for
from .diagnostics import QuantizationObserver
from .full_config import training_config
from .full_run import atomic_save, cpu_tree, get_rng, restore_rng, sha_config, train_gpu_epoch
from .full_validation import validation_reproduction, verify_checkpoint
from .glora import (ARCHITECTURES, REFERENCE_ARCH, backbone_digest, build_glora, compact_model,
                    glora_optimizer, layers, model_counts, set_subnet, subnet_parameters)
from .glora_search import evolutionary_search
from .lora_run import require_device, study_data
from .models import state_digest
from .run import ROOT, evaluate, provenance, seed_training, write_json


def read(path):
    return json.loads(Path(path).read_text())


def directory_for(root, config, seed):
    return root/config['quantization_mode']/'MNIST/glora'/f'seed_{seed}'


def fit_supernet(root, config, seed, data, device):
    directory=directory_for(root,config,seed)
    directory.mkdir(parents=True,exist_ok=True)
    config_hash=sha_config(config)
    if (directory/'fit.json').exists():
        result=read(directory/'fit.json')
        if result['config_sha256']!=config_hash:raise ValueError('Completed fit configuration changed')
        return result
    model=build_glora(config,seed,data)
    initial=dict(common_backbone_sha256=backbone_digest(model),initial_state_sha256=state_digest(model,'all'),
                 supernet_counts=model_counts(model))
    reference=read(root/'reference'/config['quantization_mode']/f'seed_{seed}'/'initialization.json')
    assert initial['common_backbone_sha256']==reference['common_backbone_sha256']
    if (directory/'initialization.json').exists():
        assert read(directory/'initialization.json')==initial
    write_json(directory/'initialization.json',initial)
    write_json(directory/'config.json',dict(config,seed=seed))
    model.to(device)
    optimizer=glora_optimizer(model,config)
    groups=[dict(name=g['name'],lr=g['lr'],weight_decay=g['weight_decay'],parameters=sum(p.numel() for p in g['params'])) for g in optimizer.param_groups]
    assert [g for g in groups if g['name']!='glora']==read(root/'reference'/config['quantization_mode']/f'seed_{seed}'/'optimizer_groups.json')
    write_json(directory/'optimizer_groups.json',groups)
    scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,mode='min',factor=.5,patience=5,threshold=.0001)
    loaders=loaders_for(data,config,seed)
    legacy=training_config(config)
    seed_training(seed+300_000)
    history,best,elapsed,peak=[],None,0.,0
    last_path=directory/'last.pt'
    if last_path.exists():
        last=torch.load(last_path,map_location='cpu',weights_only=False)
        if last['config_sha256']!=config_hash:raise ValueError('Resume configuration changed')
        model.load_state_dict(last['model']);optimizer.load_state_dict(last['optimizer']);scheduler.load_state_dict(last['scheduler'])
        history,best,elapsed,peak=last['history'],last['best'],last['training_seconds'],last['peak_cuda_memory_bytes']
        restore_rng(last['rng'],loaders,device)
        model._glora_rng.setstate(last['glora_rng'])
        atomic_save(best,directory/'best.pt')
        (directory/'epochs.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in history))
    torch.cuda.reset_peak_memory_stats(device)
    reference_architecture=[REFERENCE_ARCH]*len(layers(model))
    set_subnet(model,reference_architecture)
    for epoch in range(len(history)+1,config['epochs']+1):
        torch.cuda.synchronize(device);started=time.perf_counter()
        training=train_gpu_epoch(model,loaders['train'],optimizer,config,device)
        # Eval uses the same fixed reference architecture at every epoch, never random paths.
        validation=evaluate(model,loaders['val'],legacy,device,'val')
        if not torch.isfinite(torch.tensor(validation['loss'])):raise FloatingPointError('Non-finite validation CE')
        selected=best is None or validation['loss']<best['validation_loss']
        if selected:best=dict(epoch=epoch,validation_loss=validation['loss'],model=cpu_tree(model.state_dict()))
        scheduler.step(validation['loss'])
        torch.cuda.synchronize(device);seconds=time.perf_counter()-started
        elapsed+=seconds;peak=max(peak,torch.cuda.max_memory_allocated(device))
        history.append(dict(epoch=epoch,train=training,validation=validation,selected=selected,seconds=seconds,
                            learning_rates={g['name']:g['lr'] for g in optimizer.param_groups}))
        atomic_save(dict(config_sha256=config_hash,epoch=epoch,model=cpu_tree(model.state_dict()),
            optimizer=cpu_tree(optimizer.state_dict()),scheduler=scheduler.state_dict(),history=history,best=best,
            rng=get_rng(loaders,device),glora_rng=model._glora_rng.getstate(),training_seconds=elapsed,
            peak_cuda_memory_bytes=peak),last_path)
        if selected:atomic_save(best,directory/'best.pt')
        with (directory/'epochs.jsonl').open('a') as stream:stream.write(json.dumps(history[-1],allow_nan=False)+'\n')
        write_json(directory/'progress.json',dict(phase='supernet',epoch=epoch,epochs=config['epochs'],best_epoch=best['epoch'],
                                                validation=validation,training_seconds=elapsed))
        if epoch==1 or epoch%10==0 or epoch==config['epochs']:
            print(f"{config['quantization_mode']} GLoRA seed={seed} epoch={epoch}/{config['epochs']} val_CE={validation['loss']:.5f}",flush=True)
    verify_checkpoint(directory,best,config_hash)
    result=dict(config_sha256=config_hash,seed=seed,epochs=config['epochs'],best_epoch=best['epoch'],
        reference_validation_loss=best['validation_loss'],reference_architecture=reference_architecture,
        training_seconds=elapsed,peak_training_cuda_bytes=peak,device=str(device),gpu_name=torch.cuda.get_device_name(device),
        physical_gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),**initial)
    write_json(directory/'fit.json',result)
    return result


def search_and_export(root,config,seed,data,device):
    directory=directory_for(root,config,seed)
    if (directory/'result.json').exists():
        result=read(directory/'result.json')
        if result['config_sha256']!=sha_config(config):raise ValueError('Completed search configuration changed')
        return result
    fit=read(directory/'fit.json')
    checkpoint=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    verify_checkpoint(directory,checkpoint,sha_config(config))
    model=build_glora(config,seed,data).to(device)
    model.load_state_dict(checkpoint['model']);model.eval()
    loaders=loaders_for(data,config,seed);legacy=training_config(config)
    reference=evaluate(model,loaders['val'],legacy,device,'val')
    validation_reproduction(checkpoint['validation_loss'],reference['loss'])
    state_path=directory/'search.pt'
    started=time.perf_counter()
    torch.cuda.reset_peak_memory_stats(device)
    timing_path=directory/'search_timing.json'
    timing=read(timing_path) if timing_path.exists() else dict(validation_seconds=0.,evaluations=0,peak_cuda_bytes=0)
    def assess(candidate):
        evaluation_start=time.perf_counter()
        set_subnet(model,candidate)
        metrics=evaluate(model,loaders['val'],legacy,device,'val')
        timing['validation_seconds']+=time.perf_counter()-evaluation_start
        timing['evaluations']+=1
        timing['peak_cuda_bytes']=max(timing['peak_cuda_bytes'],torch.cuda.max_memory_allocated(device))
        write_json(timing_path,timing)
        # Atomic progress only; no test loader is evaluated during search.
        write_json(directory/'search_progress.json',dict(phase='evolution',last_validation=metrics,
                                                        session_seconds=time.perf_counter()-started))
        return metrics
    search=evolutionary_search(assess,lambda c:subnet_parameters(model,c),len(layers(model)),seed,state_path,
        population=config['search_population'],generations=config['search_generations'],elite=config['search_elite'],
        mutation_probability=config['search_mutation'],max_parameters=config['max_adapter_parameters'],
        objective_id=sha_config(config)+':'+hashlib.sha256((directory/'best.pt').read_bytes()).hexdigest())
    best=search['best'];candidate=best['architecture']
    set_subnet(model,candidate)
    compact=compact_model(model).eval()
    counts=model_counts(compact)
    assert counts['adapter_parameters']==best['parameters']
    # Validate the physically compact model, with quantization observation, before test release.
    with QuantizationObserver(compact) as observer:
        validation=evaluate(compact,loaders['val'],legacy,device,'val')
    replay=validation_reproduction(best['validation']['loss'],validation['loss'])
    quantization=observer.result()
    atomic_save(dict(config_sha256=sha_config(config),architecture=candidate,epoch=checkpoint['epoch'],
                     model=cpu_tree(compact.state_dict()),counts=counts),directory/'compact.pt')
    write_json(directory/'selected_architecture.json',dict(architecture=candidate,
        paths={name:dict(zip('ABCDE',ARCHITECTURES[index])) for (name,_),index in zip(layers(model),candidate)},
        validation=best['validation'],active_parameters=best['parameters'],search_evaluated=search['evaluated']))
    write_json(directory/'quantization.json',quantization)
    write_json(directory/'search_summary.json',search)
    result=dict(**fit,mode=config['quantization_mode'],variant='glora',validation=validation,validation_reproduction=replay,
        compact_counts=counts,active_adapter_parameters=best['parameters'],active_adapter_fp32_bytes=4*best['parameters'],
        search_evaluations=search['evaluated'],search_seconds_current_session=time.perf_counter()-started,
        search_validation_seconds=timing['validation_seconds'],search_validation_calls=timing['evaluations'],
        peak_search_cuda_bytes=max(timing['peak_cuda_bytes'],torch.cuda.max_memory_allocated(device)),
        mean_activation_effective_bits=quantization['mean_activation_effective_bits'],test=None,test_evaluations=0)
    write_json(directory/'result.json',result)
    print(f"SEARCH DONE {config['quantization_mode']} seed={seed} active_parameters={best['parameters']} val_CE={validation['loss']:.5f}",flush=True)
    return result


def verify_compact(directory,config,seed,data):
    checkpoint=torch.load(directory/'best.pt',map_location='cpu',weights_only=False)
    verify_checkpoint(directory,checkpoint,sha_config(config))
    saved=torch.load(directory/'compact.pt',map_location='cpu',weights_only=False)
    assert saved['config_sha256']==sha_config(config) and saved['epoch']==checkpoint['epoch']
    search=read(directory/'search_summary.json')
    assert search['complete'] and search['best']['architecture']==saved['architecture']
    state=torch.load(directory/'search.pt',map_location='cpu',weights_only=False)
    winner=min(state['records'],key=lambda r:(r['validation']['loss'],r['order']))
    assert winner==search['best'] and state['complete']
    model=build_glora(config,seed,data)
    model.load_state_dict(checkpoint['model']);set_subnet(model,saved['architecture'])
    compact=compact_model(model)
    expected=compact.state_dict()
    assert expected.keys()==saved['model'].keys()
    assert all(torch.equal(t,saved['model'][n]) for n,t in expected.items())
    assert model_counts(compact)==saved['counts']
    return compact,saved


def release_tests(root,configs,data):
    for config in configs:
        for seed in config['seeds']:
            directory=directory_for(root,config,seed);result=read(directory/'result.json')
            assert result['config_sha256']==sha_config(config) and result['epochs']==config['epochs']
            reference=read(root/'reference'/config['quantization_mode']/f'seed_{seed}'/'initialization.json')
            assert result['common_backbone_sha256']==reference['common_backbone_sha256']
            verify_compact(directory,config,seed,data)
    write_json(root/'test_release.json',dict(config_sha256={c['quantization_mode']:sha_config(c) for c in configs},
                                           released_unix=time.time()))


def test_once(root,config,seed,data,device):
    release=read(root/'test_release.json')
    assert release['config_sha256'][config['quantization_mode']]==sha_config(config)
    directory=directory_for(root,config,seed);result=read(directory/'result.json')
    assert result['config_sha256']==sha_config(config)
    if result['test_evaluations']:return result
    marker=directory/'test_started.json'
    if marker.exists():raise RuntimeError(f'Interrupted test requires inspection: {directory}')
    model,saved=verify_compact(directory,config,seed,data)
    model.to(device).eval()
    loader=loaders_for(data,config,seed)['test']
    with marker.open('x') as stream:json.dump(dict(started_unix=time.time(),epoch=saved['epoch']),stream)
    result['test']=evaluate(model,loader,training_config(config),device,'test')
    result['test_evaluations']=1
    write_json(directory/'result.json',result)
    print(f"TEST {config['quantization_mode']} GLoRA seed={seed}: {result['test']['accuracy']:.4f}",flush=True)
    return result


def worker(root):
    study=read(root/'study.json');torch.set_num_threads(study['threads'])
    device=require_device(study['device']);configs=study['configs']
    data,manifest=study_data(configs[0])
    with (root/'.worker.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        for config in configs:
            mode=config['quantization_mode']
            if not study['smoke']:assert manifest==read(root/'reference'/mode/'split_manifest.json')
            write_json(root/mode/'MNIST/split_manifest.json',manifest)
        for seed in configs[0]['seeds']:
            for config in configs:
                write_json(root/'active.json',dict(phase='supernet',mode=config['quantization_mode'],seed=seed))
                fit_supernet(root,config,seed,data,device)
                write_json(root/'active.json',dict(phase='search',mode=config['quantization_mode'],seed=seed))
                search_and_export(root,config,seed,data,device)
        if not study['smoke']:
            release_tests(root,configs,data)
            for seed in configs[0]['seeds']:
                for config in configs:
                    write_json(root/'active.json',dict(phase='test',mode=config['quantization_mode'],seed=seed))
                    test_once(root,config,seed,data,device)
            results=[read(directory_for(root,c,s)/'result.json') for c in configs for s in c['seeds']]
            started=[read(directory_for(root,c,s)/'test_started.json')['started_unix'] for c in configs for s in c['seeds']]
            searched=[(directory_for(root,c,s)/'quantization.json').stat().st_mtime for c in configs for s in c['seeds']]
            assert all(r['test_evaluations']==1 for r in results) and max(searched)<min(started)
            write_json(root/'completion_audit.json',dict(passed=True,completed=len(results),
                compact_checkpoints_verified=True,reference_checkpoint_minima_verified=True,
                search_validation_minima_verified=True,all_searches_preceded_tests=True,
                test_evaluations_per_checkpoint=1,audited_unix=time.time()))
        write_json(root/'active.json',dict(phase='complete'))


def coordinator(root):
    from .glora_report import generate_report
    with (root/'.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        saved=read(root/'provenance.json')
        for name,digest in saved['source_sha256'].items():
            assert hashlib.sha256((root/'source_snapshot'/name).read_bytes()).hexdigest()==digest,name
        for name,digest in read(root/'reference_hashes.json').items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest()==digest,name
        with (root/'worker.log').open('a') as stream:
            process=subprocess.Popen([sys.executable,'-B','-m','runs.lorap_study.glora_run','--worker-root',str(root)],
                cwd=root/'source_snapshot',stdout=stream,stderr=subprocess.STDOUT)
            write_json(root/'worker.json',dict(pid=process.pid,started_unix=time.time()))
            try:
                while process.poll() is None:
                    generate_report(root);time.sleep(10)
                generate_report(root)
                if process.returncode:raise RuntimeError(f'GLoRA worker failed ({process.returncode}); inspect worker.log')
            finally:
                if process.poll() is None:process.terminate()
                process.wait()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--reference-root',type=Path,default=ROOT/'result/lorap_study/table_v_control_full_20260915_v1')
    parser.add_argument('--modes',nargs='+',choices=('fixed4','a2q'),default=['fixed4','a2q'])
    parser.add_argument('--seeds',nargs='+',type=int)
    parser.add_argument('--epochs',type=int)
    parser.add_argument('--rank',type=int,default=2)
    parser.add_argument('--search-population',type=int,default=50)
    parser.add_argument('--search-generations',type=int,default=20)
    parser.add_argument('--search-elite',type=int,default=10)
    parser.add_argument('--search-mutation',type=float,default=.2)
    parser.add_argument('--max-adapter-parameters',type=int)
    parser.add_argument('--device',default='cuda:0')
    parser.add_argument('--smoke',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--worker-root',type=Path,help=argparse.SUPPRESS)
    parser.add_argument('--coordinator-root',type=Path,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    if args.worker_root:worker(args.worker_root);return
    if args.coordinator_root:coordinator(args.coordinator_root);return
    if args.output_dir is None:parser.error('--output-dir is required')
    root=args.output_dir.resolve()
    if not args.resume:
        if min(args.rank,args.search_population,args.search_generations,args.search_elite,
               1 if args.epochs is None else args.epochs)<1:
            parser.error('Rank, epochs and search sizes must be positive')
        if args.search_elite>args.search_population or not 0<=args.search_mutation<=1:
            parser.error('Invalid evolution parameters')
        if args.max_adapter_parameters is not None and args.max_adapter_parameters<0:parser.error('Negative parameter budget')
        if len(set(args.modes))!=len(args.modes) or args.seeds and len(set(args.seeds))!=len(args.seeds):parser.error('Duplicate modes/seeds')
        require_device(args.device)
        reference=args.reference_root.resolve()
        assert read(reference/'status.json')['complete']
        suite=read(reference/'suite.json');torch.set_num_threads(suite['threads'])
        seeds=args.seeds or suite['seeds']
        if args.smoke:seeds=seeds[:1]
        if not set(seeds)<=set(suite['seeds']):parser.error('Seeds must have completed matching no-prompt controls')
        # Reject accidental changes to the model/training dependencies used by the matched controls.
        old_provenance=read(reference/'provenance.json')
        original_versions=old_provenance.get('original_provenance',old_provenance)['versions']
        for package,version in original_versions.items():
            assert importlib.metadata.version(package)==version,f'Dependency changed: {package}'
        for name,digest in old_provenance['source_sha256'].items():
            if not (ROOT/name).exists() or hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest:
                raise ValueError(f'Reference implementation changed: {name}')
        root.mkdir(parents=True,exist_ok=False)
        configs=[];reference_hashes={};initializations=[]
        for mode in args.modes:
            original=read(reference/mode/'MNIST/config.json')
            config=dict(original,variants=['glora'],profile='glora_joint_qat',glora_rank=args.rank,seeds=seeds,
                epochs=2 if args.smoke else (args.epochs or original['epochs']),
                subset_sizes=dict(train=128,val=64,test=64) if args.smoke else original['subset_sizes'],
                search_population=4 if args.smoke else args.search_population,
                search_generations=2 if args.smoke else args.search_generations,
                search_elite=2 if args.smoke else args.search_elite,search_mutation=args.search_mutation,
                max_adapter_parameters=args.max_adapter_parameters)
            configs.append(config);write_json(root/mode/'MNIST/config.json',config)
            write_json(root/mode/'protocol_diff.json',{k:dict(reference=original.get(k),glora=v) for k,v in config.items() if original.get(k)!=v})
            sources=[(reference/mode/'MNIST/split_manifest.json',root/'reference'/mode/'split_manifest.json')]
            for seed in seeds:
                for filename in ('result.json','initialization.json','optimizer_groups.json'):
                    sources.append((reference/mode/'MNIST/no_prompt'/f'seed_{seed}'/filename,root/'reference'/mode/f'seed_{seed}'/filename))
                sources.append((reference/'reference'/mode/'MNIST/lora'/f'seed_{seed}'/'result.json',root/'reference'/mode/f'seed_{seed}'/'lora_result.json'))
            for source,destination in sources:
                destination.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,destination)
                reference_hashes[str(destination.relative_to(root))]=hashlib.sha256(destination.read_bytes()).hexdigest()
            for seed in seeds:
                model=build_glora(config,seed,{'train':[Data(x=torch.zeros(1,1),pos=torch.zeros(1,2))]})
                fingerprint=backbone_digest(model)
                assert fingerprint==read(root/'reference'/mode/f'seed_{seed}'/'initialization.json')['common_backbone_sha256']
                counts=model_counts(model)
                assert counts['external_prompt_parameters']==0
                initializations.append(dict(mode=mode,seed=seed,common_backbone_sha256=fingerprint,**counts))
        write_json(root/'reference_hashes.json',reference_hashes)
        write_json(root/'pairing_preflight.json',dict(passed=True,checks=initializations))
        write_json(root/'study.json',dict(configs=configs,device=args.device,threads=suite['threads'],smoke=args.smoke,
            method='GLoRA joint-QAT',reference_root=str(reference),protocol_version=1))
        provenance(root)
        # Official upstream source is archived as evidence and for parity tests, never used as a training entrypoint.
        official=ROOT/'runs/lorap_study/glora_official'
        destination=root/'source_snapshot/runs/lorap_study/glora_official'
        shutil.copytree(official,destination)
        method_notes=ROOT/'runs/lorap_study/GLORA_METHOD.md'
        archived_notes=root/'source_snapshot/runs/lorap_study/GLORA_METHOD.md'
        shutil.copy2(method_notes,archived_notes)
        recorded=read(root/'provenance.json')
        recorded['source_sha256'][str(archived_notes.relative_to(root/'source_snapshot'))]=hashlib.sha256(archived_notes.read_bytes()).hexdigest()
        for path in sorted(destination.rglob('*')):
            if path.is_file():recorded['source_sha256'][str(path.relative_to(root/'source_snapshot'))]=hashlib.sha256(path.read_bytes()).hexdigest()
        recorded['method_source']=read(official/'manifest.json')
        write_json(root/'provenance.json',recorded)
        (root/'RUN_COMMANDS.md').write_text(f'{sys.executable} -B -m runs.lorap_study.glora_run --resume --output-dir {root}\n')
    elif not (root/'study.json').exists():parser.error('No saved GLoRA study to resume')
    subprocess.run([sys.executable,'-B','-m','runs.lorap_study.glora_run','--coordinator-root',str(root)],
                   cwd=root/'source_snapshot',check=True)


if __name__=='__main__':main()
