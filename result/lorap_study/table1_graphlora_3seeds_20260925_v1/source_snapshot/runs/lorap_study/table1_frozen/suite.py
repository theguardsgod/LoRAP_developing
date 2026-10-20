"""Durable local GPU queue. Each job is a separate, resumable process.

No external scheduler, network service, paid resource, or telemetry is used.
Two workers claim files atomically and never touch unrelated GPU processes.
"""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path
from .registry import CELLS,METHODS,candidates

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]


def write(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n');os.replace(temporary,path)


def jobs(root):
    root=Path(root)
    # Order all baseline cells before adaptation; workers may adapt completed
    # cells while the final long backbone is still occupied on the other GPU.
    for name,c in CELLS.items():
        yield dict(id=name+'_s41_baseline',cell=name,stage='baseline',seed=41,
                   done=root/name/'seed_41/baseline/done.json',requires=[],
                   args=['train','--cells',name,'--seeds','41'])
    for name,c in CELLS.items():
        for method in METHODS:
            for hp in candidates(method):
                yield dict(id=f"{name}_s41_{method}_c{hp['candidate']}",cell=name,stage='tune',seed=41,
                    done=root/name/'seed_41'/method/f"candidate_{hp['candidate']}"/'done.json',
                    requires=[root/name/'seed_41/baseline/done.json'],
                    args=['adapt','--cells',name,'--seed','41','--methods',method,'--candidate',str(hp['candidate'])])
    locked=root/'SELECTION_LOCK.json'
    if locked.exists():
        for seed in (42,43):
            for name,c in CELLS.items():
                yield dict(id=f'{name}_s{seed}_baseline',cell=name,stage='baseline',seed=seed,
                           done=root/name/f'seed_{seed}/baseline/done.json',requires=[locked],
                           args=['train','--cells',name,'--seeds',str(seed)])
            for name,c in CELLS.items():
                selected=json.loads((root/name/'selection.json').read_text())['selection']
                for method in METHODS:
                    hp=selected[method]['hyperparameters']
                    yield dict(id=f"{name}_s{seed}_{method}_c{hp['candidate']}",cell=name,stage='confirmation',seed=seed,
                        done=root/name/f'seed_{seed}'/method/f"candidate_{hp['candidate']}"/'done.json',
                        requires=[locked,root/name/f'seed_{seed}/baseline/done.json'],
                        args=['adapt','--cells',name,'--seed',str(seed),'--methods',method])


def lock_selection(root):
    root=Path(root)
    if (root/'SELECTION_LOCK.json').exists():return True
    cells={}
    for name in CELLS:
        methods={}
        for method in METHODS:
            rows=[]
            for hp in candidates(method):
                path=root/name/'seed_41'/method/f"candidate_{hp['candidate']}"/'done.json'
                if not path.exists():return False
                row=json.loads(path.read_text())
                if not row['full_split'] or row['test_evaluations']!=0:raise ValueError('Invalid tuning result')
                rows.append(row)
            best=min(rows,key=lambda r:(r['validation']['loss'],r['hyperparameters']['candidate']))
            methods[method]=dict(hyperparameters=best['hyperparameters'],validation=best['validation'],source_config_hash=best['config_hash'])
        cells[name]=dict(cell=name,selection=methods,criterion='validation loss',test_evaluations=0)
    for name,record in cells.items():write(root/name/'selection.json',record)
    payload=json.dumps(cells,sort_keys=True)
    write(root/'SELECTION_LOCK.json',dict(cells=cells,sha256=hashlib.sha256(payload.encode()).hexdigest(),
                                         created=time.time(),test_evaluations=0))
    return True


def pid_alive(pid):
    try:os.kill(pid,0);return True
    except ProcessLookupError:return False


def status(root):
    root=Path(root);todo=list(jobs(root));completed=[j for j in todo if j['done'].exists()]
    failed=list((root/'queue/failures').glob('*.json')) if (root/'queue/failures').exists() else []
    running=[]
    for path in (root/'queue/claims').glob('*.json'):
        try:
            row=json.loads(path.read_text())
            if pid_alive(row['worker_pid']):running.append(row)
        except (FileNotFoundError,json.JSONDecodeError):pass
    return dict(known_jobs=len(todo),completed_jobs=len(completed),
                planned_total_training_jobs=45*(3+sum(len(candidates(m)) for m in METHODS)+2*len(METHODS)),
                completed_baselines=sum(j['stage']=='baseline' for j in completed),
                completed_tuning=sum(j['stage']=='tune' for j in completed),
                completed_confirmation=sum(j['stage']=='confirmation' for j in completed),
                failed_jobs=[p.stem for p in failed],running=running,
                selection_locked=(root/'SELECTION_LOCK.json').exists(),
                final_report_ready=(root/'FINAL_REPORT.md').exists(),timestamp=time.time())


def worker(root,device):
    root=Path(root).resolve();queue=root/'queue';queue.mkdir(parents=True,exist_ok=True)
    for sub in ('claims','failures','logs'): (queue/sub).mkdir(exist_ok=True)
    # An OS lock is released automatically after a crash and prevents duplicate
    # workers on the same GPU in this study.
    guard=open(queue/(device.replace(':','_')+'.lock'),'a')
    fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stop=False
    def request_stop(signum,frame):
        nonlocal stop
        stop=True
    signal.signal(signal.SIGTERM,request_stop)
    while not stop and not (queue/'STOP').exists():
        lock_selection(root);chosen=None
        for job in jobs(root):
            if job['done'].exists() or not all(p.exists() for p in job['requires']):continue
            if (queue/'failures'/f"{job['id']}.json").exists():continue
            claim=queue/'claims'/f"{job['id']}.json"
            if claim.exists():
                try:
                    previous=json.loads(claim.read_text())
                    if pid_alive(previous['worker_pid']):continue
                    if previous.get('child_pid') and pid_alive(previous['child_pid']):continue
                    claim.unlink()
                except (FileNotFoundError,json.JSONDecodeError):continue
            try:
                fd=os.open(claim,os.O_WRONLY|os.O_CREAT|os.O_EXCL)
            except FileExistsError:continue
            record=dict(job=job['id'],cell=job['cell'],stage=job['stage'],seed=job['seed'],device=device,worker_pid=os.getpid(),started=time.time())
            with os.fdopen(fd,'w') as file:json.dump(record,file)
            chosen=(job,claim,record);break
        if chosen is None:
            progress=status(root);write(root/'progress.json',progress)
            if progress['selection_locked'] and progress['completed_jobs']==progress['planned_total_training_jobs']:
                (root/'TRAINING_COMPLETE').touch();return
            if not progress['running'] and progress['failed_jobs']:return
            time.sleep(10);continue
        job,claim,record=chosen
        log=queue/'logs'/f"{job['id']}.log"
        command=[sys.executable,'-B','-m','runs.lorap_study.table1_frozen.'+job['args'][0],
                 '--root',str(root),'--device',device,*job['args'][1:]]
        env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='4')
        with log.open('a') as output:
            output.write('\nRUN '+json.dumps(command)+'\n');output.flush()
            child=subprocess.Popen(command,cwd=REPO,stdout=output,stderr=subprocess.STDOUT,env=env)
            record['child_pid']=child.pid;write(claim,record)
            write(root/'progress.json',status(root))
            code=child.wait()
        if code or not job['done'].exists():
            write(queue/'failures'/f"{job['id']}.json",dict(record,returncode=code,log=str(log),finished=time.time()))
        claim.unlink(missing_ok=True);write(root/'progress.json',status(root))


def main():
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['worker','status','manifest'])
    parser.add_argument('--root',required=True);parser.add_argument('--device',default='cuda:2')
    args=parser.parse_args()
    if args.action=='worker':worker(args.root,args.device)
    elif args.action=='status':print(json.dumps(status(args.root),ensure_ascii=False,indent=2))
    else:
        write(Path(args.root)/'manifest.json',dict(cells=CELLS,methods=METHODS,seeds=[41,42,43],
                candidate_grid={m:candidates(m) for m in METHODS},gpu_only=True,selection='validation loss',test_gate='locked'))
        print('45-cell manifest written')


if __name__=='__main__':main()
