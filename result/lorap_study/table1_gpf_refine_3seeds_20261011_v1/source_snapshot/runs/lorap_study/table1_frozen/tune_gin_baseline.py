"""Bounded validation-only search, authorized by GIN_BASELINE_SEARCH.md."""
import argparse,json,shutil
from pathlib import Path
import torch
from .registry import CELLS,METHODS
from .train import baseline
from .common import write_json,file_lock


OPTIONS=[dict(lr=.01,weight_decay=.02),dict(lr=.01,weight_decay=.05),
         dict(lr=.01,weight_decay=.1),dict(lr=.01,weight_decay=.2),dict(lr=.005,weight_decay=.05)]


def search(root,name,device):
    root=Path(root).resolve();cell=root/name;c=CELLS[name];folder=cell/'baseline_search';folder.mkdir(exist_ok=True)
    with file_lock(folder/'search.lock'):
        if (folder/'selection.json').exists():return
        original=cell/'seed_41/baseline';done=json.loads((original/'done.json').read_text())
        history=json.loads((original/'history.json').read_text())
        gap=max(r['val']['metric'] for r in history)-done['validation']['metric']
        if gap<=10:raise ValueError('Search trigger not met')
        records=[dict(candidate=0,hyperparameters=dict(lr=c['lr'],weight_decay=c['weight_decay']),
                      validation=done['validation'],source=str(original),config_hash=done['config_hash'])]
        for i,hp in enumerate(OPTIONS,1):
            result=baseline(folder/f'candidate_{i}',dict(c,**hp),41,device)
            records.append(dict(candidate=i,hyperparameters=hp,validation=result['validation'],
                 source=str(folder/f'candidate_{i}'/name/'seed_41/baseline'),config_hash=result['config_hash']))
            write_json(folder/'candidates.json',records)
        selected=min(records,key=lambda r:(r['validation']['loss'],r['candidate']))
        # No method may have started on the previous backbone, including an
        # interrupted fit. This prevents mixing paired comparison populations.
        seedpath=cell/'seed_41'
        if (seedpath/'calibration.pt').exists() or any((seedpath/m).exists() for m in METHODS):
            raise RuntimeError('Frozen adaptation has already started; cannot replace its backbone')
        if (root/'SELECTION_LOCK.json').exists():raise RuntimeError('Selection already locked')
        if selected['candidate']:
            if (folder/'native').exists():raise RuntimeError('Native backup already exists; audit interrupted promotion')
            shutil.copytree(selected['source'],seedpath/'baseline_pending')
            original.rename(folder/'native');(seedpath/'baseline_pending').rename(original)
            records[0]['source']=str(folder/'native')
            write_json(cell/'baseline_override.json',dict(hyperparameters=selected['hyperparameters'],
                       criterion='minimum validation loss',candidate=selected['candidate'],
                       validation=selected['validation'],scope='all paired seeds, fixed before adapters'))
        write_json(folder/'selection.json',dict(trigger_gap=gap,selected=selected,candidates=records,
                   test_evaluations=0,quantization_budget_changed=False))
        print(name,'selected baseline',selected['candidate'],selected['validation'],flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--device',default='cuda:2')
    a=p.parse_args();torch.set_num_threads(4)
    for name in ('A2Q_Cora_GIN','A2Q_CiteSeer_GIN'):search(a.root,name,a.device)


if __name__=='__main__':main()
