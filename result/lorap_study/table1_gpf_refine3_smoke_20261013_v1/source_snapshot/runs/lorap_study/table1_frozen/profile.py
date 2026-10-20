"""Small training-only throughput measurement; never an accuracy experiment."""
import argparse,time
import torch
from .registry import CELLS
from .data import Bundle,labels_and_output
from .backends import build,predict,optimizer,project
from .train import task_loss
from .common import write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--cells',nargs='+',required=True);p.add_argument('--device',default='cuda:2');p.add_argument('--out',required=True)
    args=p.parse_args();torch.set_num_threads(4);rows=[]
    for name in args.cells:
        c=CELLS[name];b=Bundle(c);model=build(c,b,41).to(args.device).train();opt=optimizer(model,c)
        torch.cuda.reset_peak_memory_stats(args.device);times=[]
        start=time.monotonic()
        for i,batch in enumerate(b.batches('train',41,0,args.device)):
            opt.zero_grad(set_to_none=True);out,budget=predict(model,c,batch)
            logits,y=labels_and_output(c,batch,out,'train');loss=task_loss(c,logits,y)+c['a_loss']*torch.relu(budget-c['a_storage']).square()
            loss.backward();opt.step();project(model,c);torch.cuda.synchronize(args.device)
            times.append(time.monotonic()-start);start=time.monotonic()
            if i==5:break
        row=dict(cell=name,batch_size=c['batch_size'],seconds_per_batch=sum(times[1:])/len(times[1:]),
                 estimated_train_epoch_seconds=sum(times[1:])/len(times[1:])*((len(b.train)+c['batch_size']-1)//c['batch_size']),
                 peak_mib=torch.cuda.max_memory_allocated(args.device)/(1024**2))
        rows.append(row);print(row,flush=True);write_json(args.out,rows)
        del b,model,opt,out,logits,loss,batch;torch.cuda.empty_cache()


if __name__=='__main__':main()
