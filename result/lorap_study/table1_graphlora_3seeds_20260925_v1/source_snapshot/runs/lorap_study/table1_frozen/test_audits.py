"""Independent local-target arithmetic, deployment and test-gate checks."""
import json,os,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
import torch
from torch_geometric.data import Data
from .registry import CELLS,candidates
from .backends import build,predict,load_state
from .instrument import Instrument
from .adapters import FrozenModel,calibrate
from .data import Bundle
from .common import digest,cpu_tree
from .deploy import export,load
from .finish import test_manifest


class AuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device=os.environ.get('TABLE1_DEVICE','cuda:2');torch.set_num_threads(4)
        if not torch.cuda.is_available():raise RuntimeError('GPU is mandatory')

    def test_reference_is_message_only(self):
        for framework in ('A2Q','QAT','DQ'):
            with self.subTest(framework=framework):
                c=dict(CELLS[framework+'_Cora_GIN'])
                b=SimpleNamespace(data=Data(num_nodes=4),features=3,classes=2)
                model=build(c,b,41).to(self.device)
                layer=model.convs[0] if framework=='A2Q' else model.convs[1]
                x=torch.arange(64,device=self.device,dtype=torch.float32).reshape(4,16)/17
                edges=torch.tensor([[0,0,1,2],[1,2,2,1]],device=self.device)
                mask=torch.zeros(4,dtype=torch.bool,device=self.device)
                def call():
                    return layer(x,edges,mask) if framework=='DQ' else layer(x,edges)
                model.train()
                with torch.no_grad():call()
                model.eval();captured={}
                qm=layer.q_xw if framework=='A2Q' else layer.mp_quantizers['message_low' if framework=='DQ' else 'message']
                def capture(m,args,out):captured.update(pre=args[0].detach(),q=out.detach())
                handle=qm.register_forward_hook(capture)
                def callback(i,h,ctx,target):
                    captured.update(aggregate=h.detach(),target=target,context=ctx);return h
                instrument=Instrument(model,c,callback);instrument.reset(True)
                with torch.no_grad():call()
                if framework=='A2Q':
                    raw_messages=x[edges[0]];quantized=captured['q'].T[edges[0]]
                else:raw_messages=captured['pre'];quantized=captured['q']
                raw=torch.stack([raw_messages[edges[1]==j].sum(0) for j in range(4)])
                qsum=torch.stack([quantized[edges[1]==j].sum(0) for j in range(4)])
                if framework!='A2Q':
                    qa=layer.mp_quantizers['aggregate_low' if framework=='DQ' else 'aggregate']
                    with torch.no_grad():raw=qa.forward(raw);qsum=qa.forward(qsum)
                expected=raw.double()-qsum.double()
                self.assertTrue(torch.equal(captured['target'],expected))
                self.assertTrue(torch.equal(captured['aggregate'],qsum))
                self.assertTrue(torch.equal(captured['target'][[0,3]],torch.zeros_like(expected[[0,3]])))
                self.assertFalse(captured['target'].requires_grad)
                self.assertFalse(captured['context'].requires_grad)
                instrument.reset(False)
                with torch.no_grad():call()
                self.assertIsNone(captured['target'])
                instrument.remove();handle.remove()

    def test_deploy_roundtrip_and_gate_freeze(self):
        for name in ('A2Q_Cora_GAT','DQ_MNIST_GCN','QAT_ZINC_GIN'):
            c=dict(CELLS[name],batch_size=2);b=Bundle(c);base=build(c,b,41).to(self.device)
            batch=next(b.batches('train',41,0,self.device,limit=2))
            base.train()
            with torch.no_grad():predict(base,c,batch)
            base.eval();stats=calibrate(base,c,b,self.device,41,max_graphs=2);state=cpu_tree(base.state_dict())
            for method in ('none','gpf','spectral','error','weight_lora','gate_only'):
                with self.subTest(cell=name,method=method):
                    native=build(c,b,41).to(self.device);load_state(native,state)
                    hp=None if method=='none' else candidates(method)[0]
                    model=native if method=='none' else FrozenModel(native,c,stats,hp,method,41).to(self.device)
                    model.eval()
                    if method=='gate_only':
                        with torch.no_grad():
                            for factor in model.bank.factors:factor.right.fill_(.01)
                        model.bank.freeze_for_gates()
                        trainable=[n for n,p in model.named_parameters() if p.requires_grad]
                        self.assertTrue(trainable)
                        self.assertTrue(all('outer_gates' in n for n in trainable))
                        self.assertEqual(sum(p.numel() for p in model.parameters() if p.requires_grad),5*len(stats))
                    anonymous=batch.clone()
                    for key in ('y','train_mask','val_mask','test_mask','prob_mask'):
                        if key in anonymous:del anonymous[key]
                    with torch.no_grad():reference=predict(model,c,anonymous)[0]
                    with tempfile.TemporaryDirectory() as temporary:
                        path=Path(temporary)/'inference.pt';export(path,model,c,b,41,method,hp)
                        artifact=torch.load(path,map_location='cpu',weights_only=False)
                        self.assertFalse(artifact['teacher_available']);self.assertFalse(artifact['labels_in_artifact'])
                        for row in artifact['normalization']:
                            self.assertNotIn('basis',row);self.assertNotIn('error_scale',row);self.assertNotIn('target',row)
                        restored=load(path,self.device)
                        with torch.no_grad():out=restored(anonymous)
                        self.assertTrue(torch.equal(reference,out))
                        self.assertFalse(any(p.requires_grad for p in restored.parameters()))
                        if getattr(restored.body,'instrument',None):self.assertEqual(restored.body.instrument.reference_calls,0)
                        if hasattr(restored.body,'close'):restored.body.close()
                    if hasattr(model,'close'):model.close()

    def test_test_gate_requires_all_45_cells(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaisesRegex(RuntimeError,'45'):test_manifest(temporary)
            (Path(temporary)/'SELECTION_LOCK.json').write_text(json.dumps(dict(cells={},sha256='invalid')))
            with self.assertRaisesRegex(ValueError,'coverage'):test_manifest(temporary)


if __name__=='__main__':unittest.main()
