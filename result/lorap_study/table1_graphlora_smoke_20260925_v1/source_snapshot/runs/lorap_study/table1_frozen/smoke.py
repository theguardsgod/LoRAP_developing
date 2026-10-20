"""GPU-only operator smoke tests; uses training data, never produces test results."""
import argparse,json,traceback,time
from pathlib import Path
import torch
from torch.nn import functional as F
from .registry import CELLS
from .data import Bundle,labels_and_output
from .backends import build,predict,optimizer,project


def main():
    p=argparse.ArgumentParser();p.add_argument('--device',default='cuda:2');p.add_argument('--cells',nargs='*');p.add_argument('--out',required=True)
    args=p.parse_args();torch.set_num_threads(4)
    records=[]
    for name in args.cells or CELLS:
        c=dict(CELLS[name],batch_size=2)
        try:
            start=time.monotonic();b=Bundle(c)
            model=build(c,b,41).to(args.device).train();opt=optimizer(model,c)
            batch=next(b.batches('train',41,0,args.device,limit=2))
            out,budget=predict(model,c,batch)
            logits,y=labels_and_output(c,batch,out,'train')
            assert logits.size(-1)==b.classes,(logits.shape,b.classes)
            loss=F.l1_loss(logits.flatten(),y) if c['regression'] else F.cross_entropy(logits,y)
            objective=loss+c['a_loss']*F.relu(budget-c['a_storage']).square()
            objective.backward()
            assert torch.isfinite(objective)
            for n,param in model.named_parameters():
                if param.grad is not None:assert torch.isfinite(param.grad).all(),n
            opt.step();project(model,c);model.eval()
            with torch.no_grad():out,_=predict(model,c,batch)
            assert torch.isfinite(out).all()
            row=dict(cell=name,ok=True,shape=list(out.shape),train_loss=float(loss.detach()),seconds=time.monotonic()-start,
                     parameters=sum(p.numel() for p in model.parameters()))
            del batch,model,opt,out,loss,objective,b
        except Exception as error:
            row=dict(cell=name,ok=False,error=str(error),traceback=traceback.format_exc())
        records.append(row);print(json.dumps(row),flush=True)
        path=Path(args.out);path.parent.mkdir(parents=True,exist_ok=True);path.write_text(json.dumps(records,indent=2)+'\n')
        torch.cuda.empty_cache()
    if not all(r['ok'] for r in records):raise SystemExit(1)


if __name__=='__main__':main()
