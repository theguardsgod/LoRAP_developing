"""End-to-end final-stage test on synthetic data, not benchmark test splits."""
import hashlib,json,os,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import torch
from torch_geometric.data import Data
from . import finish,report
from .registry import CELLS,candidates
from .backends import build,load_state,predict
from .adapters import FrozenModel,calibrate
from .common import cpu_tree,digest,save,write_json
from .train import evaluate


class SyntheticBundle:
    def __init__(self,c):
        self.c=c;self.features=3;self.classes=2
        self.data=Data(x=torch.arange(18,dtype=torch.float32).reshape(6,3)/16,
            edge_index=torch.tensor([[0,1,1,2,2,3,3,4,4,5,5,0],[1,0,2,1,3,2,4,3,5,4,0,5]]),
            y=torch.tensor([0,1,0,1,0,1]),train_mask=torch.tensor([1,1,0,0,0,0],dtype=torch.bool),
            val_mask=torch.tensor([0,0,1,1,0,0],dtype=torch.bool),test_mask=torch.tensor([0,0,0,0,1,1],dtype=torch.bool))
    def batches(self,split,seed,epoch,device,limit=None,allow_test=False):
        if split=='test' and not allow_test:raise RuntimeError('Synthetic test gate is locked')
        yield self.data.clone().to(device)


class FinalFlowTest(unittest.TestCase):
    def test_synthetic_locked_pipeline_and_restart(self):
        device=os.environ.get('TABLE1_DEVICE','cuda:2');torch.set_num_threads(4)
        if not torch.cuda.is_available():raise RuntimeError('GPU required')
        name='SYNTHETIC_UNIT_FIXTURE';c=dict(CELLS['QAT_Cora_GIN'],cell=name)
        methods=('task','mlp');b=SyntheticBundle(c)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);choices={}
            for seed in (41,42,43):
                base=build(c,b,seed).to(device).train();batch=next(b.batches('train',seed,0,device))
                with torch.no_grad():predict(base,c,batch)
                base.eval();state=cpu_tree(base.state_dict());stats=calibrate(base,c,b,device,seed)
                prefix=root/name/f'seed_{seed}';save(prefix/'calibration.pt',dict(stats=stats))
                for method in ('none',*methods):
                    native=build(c,b,seed).to(device);load_state(native,state)
                    hp=None if method=='none' else candidates(method)[0]
                    model=native if method=='none' else FrozenModel(native,c,stats,hp,method,seed).to(device)
                    validation=evaluate(model,c,b,device,seed)
                    path=prefix/'baseline' if method=='none' else prefix/method/'candidate_0'
                    save(path/'best.pt',dict(model=model.state_dict(),validation_loss=validation['loss'],epoch=0))
                    write_json(path/'done.json',dict(full_split=True,test_evaluations=0,backbone_unchanged=True,
                        validation=validation,backbone_sha256=digest(native),checkpoint_sha256=digest(model)))
                    if method!='none':
                        choices[method]=dict(hyperparameters=hp,validation=validation,source_config_hash='synthetic')
                        model.close()
            selected=dict(cell=name,selection=choices,criterion='validation loss',test_evaluations=0)
            lock={name:selected};write_json(root/name/'selection.json',selected)
            write_json(root/'SELECTION_LOCK.json',dict(cells=lock,sha256=hashlib.sha256(json.dumps(lock,sort_keys=True).encode()).hexdigest()))
            with patch.object(finish,'CELLS',{name:c}),patch.object(finish,'METHODS',methods),\
                 patch.object(finish,'Bundle',SyntheticBundle),patch.object(report,'CELLS',{name:c}),patch.object(report,'METHODS',methods):
                finish.run(root,device)
                self.assertTrue((root/'FINAL_REPORT.md').exists())
                self.assertEqual(len(json.loads((root/'TEST_LOCK.json').read_text())['cases']),9)
                summary=json.loads((root/'summary.json').read_text())
                self.assertTrue(all(abs(row['paired_gain'])<1e-12 for row in summary))
                # Restart of a completed final stage must not execute a model.
                with patch.object(finish,'predict',side_effect=AssertionError('Unexpected repeated prediction')):
                    finish.run(root,device)
                self.assertEqual(len(list(root.glob('*/seed_*/*/**/test.json'))),9)
                # A later replacement of one canonical backbone must invalidate
                # the lock instead of silently comparing unpaired methods.
                donepath=root/name/'seed_41/task/candidate_0/done.json'
                changed=json.loads(donepath.read_text());changed['backbone_sha256']='another-backbone'
                write_json(donepath,changed)
                with self.assertRaisesRegex(ValueError,'Paired backbone mismatch'):
                    finish.test_manifest(root)


if __name__=='__main__':unittest.main()
