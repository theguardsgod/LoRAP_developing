"""All validation choices must be locked before any official test prediction."""
import argparse,hashlib,json
from pathlib import Path
import torch
from .registry import CELLS,METHODS
from .data import Bundle,labels_and_output
from .backends import build,load_state,predict
from .adapters import FrozenModel
from .common import digest,write_json,save,file_lock
from .train import task_loss,evaluate
from .deploy import export,load as load_export
from .diagnostics import diagnose,fixed_input,quantization_record


def test_manifest(root):
    root=Path(root);file=root/'SELECTION_LOCK.json'
    if not file.exists():raise RuntimeError('All 45 validation selections must be locked first')
    lock=json.loads(file.read_text())
    if set(lock['cells'])!=set(CELLS):raise ValueError('Selection coverage is incomplete')
    if hashlib.sha256(json.dumps(lock['cells'],sort_keys=True).encode()).hexdigest()!=lock['sha256']:
        raise ValueError('Selection lock digest mismatch')
    cases=[]
    for name,c in CELLS.items():
        selected=json.loads((root/name/'selection.json').read_text())
        if selected!=lock['cells'][name]:raise ValueError('Selection changed after lock')
        for seed in c['seeds']:
            prefix=root/name/f'seed_{seed}'
            baseline_done=prefix/'baseline/done.json'
            if not baseline_done.exists():raise RuntimeError(f'Incomplete paired backbone: {prefix}')
            backbone_sha256=json.loads(baseline_done.read_text())['backbone_sha256']
            for method in ('none',*METHODS):
                hp=None if method=='none' else selected['selection'][method]['hyperparameters']
                directory=prefix/'baseline' if method=='none' else prefix/method/f"candidate_{hp['candidate']}"
                if not (directory/'done.json').exists():raise RuntimeError(f'Incomplete selected run: {directory}')
                done=json.loads((directory/'done.json').read_text())
                if not done['full_split'] or done['test_evaluations']!=0:raise ValueError('Invalid formal run')
                if method!='none' and not done['backbone_unchanged']:raise ValueError('Backbone changed')
                if done['backbone_sha256']!=backbone_sha256:raise ValueError('Paired backbone mismatch')
                checkpoint=directory/'best.pt'
                cases.append(dict(cell=name,seed=seed,method=method,hyperparameters=hp,
                    directory=str(directory.relative_to(root)),checkpoint_file_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                    state_sha256=done['backbone_sha256'] if method=='none' else done['checkpoint_sha256']))
    return dict(selection_sha256=lock['sha256'],cases=cases,
                sha256=hashlib.sha256(json.dumps(cases,sort_keys=True).encode()).hexdigest())


def lock_tests(root):
    root=Path(root);manifest=test_manifest(root);path=root/'TEST_LOCK.json'
    if path.exists():
        if json.loads(path.read_text())!=manifest:raise RuntimeError('Test inputs changed after unlocking')
    else:write_json(path,manifest)
    return manifest


@torch.no_grad()
def test_once(root,case,model,c,b,device):
    root=Path(root);path=root/case['directory'];result=path/'test.json'
    if result.exists():return json.loads(result.read_text())
    manifest=json.loads((root/'TEST_LOCK.json').read_text())
    if case not in manifest['cases']:raise RuntimeError('Checkpoint not registered in test lock')
    chunkdir=path/'test_predictions';chunkdir.mkdir(exist_ok=True)
    totals=torch.zeros(2,dtype=torch.float64,device=device);count=0
    for index,batch in enumerate(b.batches('test',case['seed'],0,device,allow_test=True)):
        chunk=chunkdir/f'{index:05d}.pt'
        if chunk.exists():
            prediction=torch.load(chunk,map_location=device,weights_only=False)
            if prediction['checkpoint_file_sha256']!=case['checkpoint_file_sha256']:
                raise ValueError('Cached test prediction belongs to another checkpoint')
            out,y=prediction['logits'],prediction['labels']
        else:
            out,_=predict(model,c,batch);out,y=labels_and_output(c,batch,out,'test')
            save(chunk,dict(logits=out,labels=y,checkpoint_file_sha256=case['checkpoint_file_sha256']))
        loss=task_loss(c,out,y)
        metric=(out.flatten()-y).abs().sum() if c['regression'] else (out.argmax(1)==y).sum()
        totals+=torch.stack((loss.double()*len(y),metric.double()));count+=len(y)
    v=totals.cpu().tolist()
    record=dict(cell=case['cell'],seed=case['seed'],method=case['method'],loss=v[0]/count,
                metric=v[1]/count*(1 if c['regression'] else 100),examples=count,
                completed_test_evaluations=1,checkpoint_file_sha256=case['checkpoint_file_sha256'],test_lock_sha256=manifest['sha256'])
    write_json(result,record);return record


def run(root,device):
    root=Path(root)
    with file_lock(root/'final_evaluation.lock'):
        manifest=lock_tests(root)
        for case in manifest['cases']:
            path=root/case['directory'];c=CELLS[case['cell']];seed=case['seed'];method=case['method']
            if (path/'final_complete.json').exists():continue
            b=Bundle(c);backbone=build(c,b,seed).to(device)
            checkpoint=torch.load(path/'best.pt',map_location='cpu',weights_only=False)
            if method=='none':model=backbone;load_state(model,checkpoint['model'])
            else:
                stats=torch.load(root/case['cell']/f'seed_{seed}/calibration.pt',map_location='cpu',weights_only=False)['stats']
                model=FrozenModel(backbone,c,stats,case['hyperparameters'],method,seed).to(device)
                load_state(model,checkpoint['model'])
            model.eval()
            if digest(model)!=case['state_sha256']:raise AssertionError('Loaded state mismatch')
            done=json.loads((path/'done.json').read_text());validation=evaluate(model,c,b,device,seed)
            if abs(validation['loss']-done['validation']['loss'])>1e-6:raise AssertionError('Pretest validation replay mismatch')
            export(path/'inference.pt',model,c,b,seed,method,case['hyperparameters'])
            deployed=load_export(path/'inference.pt',device)
            batch=next(b.batches('val',seed,0,device));anonymous=batch.clone();anonymous.y=None
            with torch.no_grad():
                original=predict(model,c,batch)[0];restored=deployed(anonymous)
            if not torch.equal(original,restored):raise AssertionError('Inference export roundtrip mismatch')
            if hasattr(deployed.body,'instrument') and deployed.body.instrument:
                if deployed.body.instrument.reference_calls:raise AssertionError('Teacher called during deployment')
            if hasattr(deployed.body,'close'):deployed.body.close()
            del deployed,batch,anonymous,original,restored
            test=test_once(root,case,model,c,b,device)
            write_json(path/'quantization.json',quantization_record(backbone))
            if method!='none':
                own=diagnose(model,c,b,seed,device) if model.bank is not None else dict(applicable=False,reason='No aggregation predictor')
                common=fixed_input(model,c,b,seed,device)
                write_json(path/'diagnostics.json',dict(own_input=own,fixed_input=common))
                model.close()
            write_json(path/'final_complete.json',dict(test=test,export_roundtrip_exact=True,
                       exported_prediction_uses_teacher=False,unlabeled_inference=True))
            print('final evaluated:',case['cell'],seed,method,flush=True)
            del model,backbone,b;torch.cuda.empty_cache()
        from .report import generate
        generate(root)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--device',default='cuda:2');p.add_argument('--check-only',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    if a.check_only:print(json.dumps(test_manifest(a.root),indent=2))
    else:run(a.root,a.device)


if __name__=='__main__':main()
