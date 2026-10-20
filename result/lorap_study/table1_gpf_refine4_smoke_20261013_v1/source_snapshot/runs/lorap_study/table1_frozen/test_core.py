import json,os,unittest
from pathlib import Path
import torch
from torch.nn import functional as F
from .registry import CELLS,METHODS,candidates
from .data import Bundle,labels_and_output
from .backends import build,predict,load_state
from .adapters import FrozenModel,calibrate,mlp_widths
from .instrument import context_nodes
from .common import digest,cpu_tree,consider,write_json


class CoreTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.device=os.environ.get('TABLE1_DEVICE','cuda:2');torch.set_num_threads(4)
        if not torch.cuda.is_available():raise RuntimeError('These tests require GPU')

    def test_registry_and_ties(self):
        self.assertEqual(len(CELLS),45)
        self.assertEqual(CELLS['A2Q_Cora_GCN']['paper_none'],76.1)
        self.assertEqual(CELLS['QAT_ZINC_GAT']['paper_none'],.702)
        module=torch.nn.Linear(2,1).to(self.device)
        best=consider(None,1.,2,module)
        self.assertIs(consider(best,1.,3,module),best)
        self.assertEqual(consider(best,.9,4,module)['epoch'],4)

    def test_no_neighbors(self):
        q=torch.zeros(4,3,device=self.device)
        edges=torch.tensor([[0,1],[1,2]],device=self.device)
        ctx=context_nodes(q,edges,torch.ones(1,3,device=self.device),torch.full((1,3),4.,device=self.device))
        self.assertTrue(torch.equal(ctx[[0,3]],torch.zeros(2,4,device=self.device)))
        self.assertTrue(torch.isfinite(ctx).all())

    def test_all_backends_and_methods(self):
        names=os.environ.get('TABLE1_CELLS','').split() or list(CELLS)
        results=[]
        for name in names:
            c=dict(CELLS[name],batch_size=2);b=Bundle(c);base=build(c,b,41).to(self.device)
            batch=next(b.batches('train',41,0,self.device,limit=2))
            base.train()
            with torch.no_grad():predict(base,c,batch) # initialize native QAT/DQ observers on train only
            base.eval()
            with torch.no_grad():reference,_=predict(base,c,batch)
            state=cpu_tree(base.state_dict());base_digest=digest(base)
            stats=calibrate(base,c,b,self.device,41,max_graphs=2)
            self.assertEqual(base_digest,digest(base))
            task_state=None
            for method in METHODS:
                with self.subTest(cell=name,method=method):
                    native=build(c,b,41).to(self.device);load_state(native,state)
                    model=FrozenModel(native,c,stats,candidates(method)[0],method,41).to(self.device)
                    if method=='task':task_state=cpu_tree(model.bank.state_dict())
                    if method in ('error','gate_only'):
                        shared={n:t for n,t in model.bank.state_dict().items() if not n.startswith('outer_gates.')}
                        self.assertEqual(digest(shared),digest(task_state))
                    model.train()
                    self.assertFalse(model.backbone.training)
                    self.assertFalse(any(p.requires_grad for p in model.backbone.parameters()))
                    initial,_=model(batch)
                    if method not in ('lorap','gpf','gpf_lorap'):
                        self.assertTrue(torch.equal(initial,reference),(name,method,float((initial-reference).abs().max())))
                    if model.instrument:self.assertEqual(model.instrument.reference_calls,0)
                    if model.instrument:
                        logits,_=model(batch,teacher=True)
                        self.assertTrue(torch.equal(logits,initial),(name,method,'teacher altered predictions'))
                        self.assertTrue(all(torch.isfinite(loss) for loss in model.losses))
                    else:logits=initial
                    out,y=labels_and_output(c,batch,logits,'train')
                    loss=F.l1_loss(out.flatten(),y) if c['regression'] else F.cross_entropy(out,y)
                    if model.losses:loss=loss+.1*torch.stack(model.losses).mean()
                    loss.backward()
                    params=[p for p in model.parameters() if p.requires_grad]
                    grads=[p.grad for p in params if p.grad is not None]
                    self.assertTrue(grads)
                    self.assertTrue(all(torch.isfinite(g).all() for g in grads))
                    torch.optim.Adam(params,lr=.001).step()
                    self.assertEqual(digest(model.backbone),base_digest)
                    unlabeled=batch.clone();unlabeled.y=None
                    with torch.no_grad():
                        after,_=model(batch);anonymous,_=model(unlabeled)
                    self.assertTrue(torch.equal(after,anonymous))
                    self.assertTrue(torch.isfinite(after).all())
                    if method=='mlp':
                        self.assertTrue(all(row['gap']<=.05 for row in model.bank.parameter_matches))
                    results.append(dict(cell=name,method=method,ok=True,trainable=sum(p.numel() for p in params)))
                    model.close();del model,native,loss,logits,initial,after,anonymous,params,grads
            del base,batch,b
            print('adapter audit passed:',name,flush=True)
            write_json(Path(os.environ.get('TABLE1_AUDIT_OUT','/tmp/table1_adapter_audit.json')),results)
            torch.cuda.empty_cache()


if __name__=='__main__':unittest.main()
