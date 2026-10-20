"""Atomic experiment records, resumable RNG state, and validation selection."""
import fcntl,hashlib,json,os,random
from contextlib import contextmanager
from pathlib import Path
import numpy as np
import torch


def cpu_tree(value):
    if torch.is_tensor(value):return value.detach().cpu().clone()
    if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return type(value)(cpu_tree(v) for v in value)
    return value


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    torch.save(cpu_tree(value),temporary);os.replace(temporary,path)


def write_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n');os.replace(temporary,path)


def digest(state):
    if hasattr(state,'state_dict'):state=state.state_dict()
    h=hashlib.sha256()
    for n,t in sorted(state.items()):
        h.update(n.encode());h.update(str(t.dtype).encode());h.update(str(tuple(t.shape)).encode())
        h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def config_hash(c):return hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()


@contextmanager
def file_lock(path):
    with open(path,'a') as file:
        fcntl.flock(file,fcntl.LOCK_EX)
        try:yield
        finally:fcntl.flock(file,fcntl.LOCK_UN)


def seed_all(seed):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)


def rng_state(device):
    return dict(python=random.getstate(),numpy=np.random.get_state(),torch=torch.get_rng_state(),cuda=torch.cuda.get_rng_state(device))


def restore_rng(state,device):
    random.setstate(state['python']);np.random.set_state(state['numpy']);torch.set_rng_state(state['torch']);torch.cuda.set_rng_state(state['cuda'],device)


def finite_gradients(model):
    checks=[torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
    if not checks or not torch.stack(checks).all():raise FloatingPointError('Missing or nonfinite gradients')


def consider(best,loss,epoch,model):
    if not np.isfinite(loss):raise FloatingPointError('Nonfinite validation loss')
    if best is None or loss<best['validation_loss']:
        return dict(validation_loss=loss,epoch=epoch,model=cpu_tree(model.state_dict()))
    return best
