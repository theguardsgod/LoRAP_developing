"""Local continuation of the same experiment: finish tests after training.

This is a dependent stage of the training program, not a scheduled Codex task.
It writes status files quietly and never sends messages or calls external apps.
"""
import argparse,json,os,subprocess,sys,time
from datetime import datetime,timezone,timedelta
from pathlib import Path
from .suite import status,write,REPO


def update_status(root):
    progress=status(root);write(root/'progress.json',progress)
    lines=['# 表 I 冻结骨干实验进度','',
           '更新时间：'+datetime.now(timezone(timedelta(hours=8))).isoformat(timespec='seconds'),'',
           f"范围：45 组 × 10 种冻结方法 × 3 个配对种子；计划 {progress['planned_total_training_jobs']} 个训练/调参任务。",'',
           f"已完成训练任务：{progress['completed_jobs']}；其中骨干 {progress['completed_baselines']}、验证调参 {progress['completed_tuning']}、后续种子确认 {progress['completed_confirmation']}。",'',
           f"验证配置锁：{'已锁定' if progress['selection_locked'] else '尚未锁定'}。最终报告：{'已生成' if progress['final_report_ready'] else '尚未完成'}。",'',
           '## 当前 GPU 任务','']
    for job in progress['running']:
        lines.append(f"- {job['device']}：{job['job']}")
    if not progress['running']:lines.append('- 当前没有可见的训练 worker 任务。')
    if progress['failed_jobs']:
        lines+=['','## 需要处理的失败','']+[f'- {name}' for name in progress['failed_jobs']]
    if not progress['final_report_ready']:
        lines+=['','当前不能判断哪些方法超过论文表 I。实现/功能检查的通过不等于测试性能提升。']
    else:lines+=['',f'[完整中文报告]({root/"FINAL_REPORT.md"})']
    (root/'STATUS.md').write_text('\n'.join(lines)+'\n')
    return progress


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--device',default='cuda:2')
    a=p.parse_args();root=Path(a.root).resolve();queue=root/'queue';queue.mkdir(exist_ok=True)
    # Avoid more than one final-stage watcher after a resume.
    import fcntl
    guard=open(queue/'pipeline.lock','a');fcntl.flock(guard,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(queue/'pipeline.json',dict(pid=os.getpid(),device=a.device,started=time.time(),stage='waiting_for_training'))
    while not (root/'TRAINING_COMPLETE').exists():
        progress=update_status(root)
        if progress['final_report_ready']:return
        if (queue/'STOP').exists():return
        if progress['failed_jobs'] and not progress['running']:
            write(queue/'needs_attention.json',progress);return
        time.sleep(30)
    write(queue/'pipeline.json',dict(pid=os.getpid(),device=a.device,stage='final_test_export_diagnostics'))
    with (queue/'final_evaluation.log').open('a') as log:
        code=subprocess.call([sys.executable,'-B','-m','runs.lorap_study.table1_frozen.finish',
                              '--root',str(root),'--device',a.device],cwd=REPO,stdout=log,stderr=subprocess.STDOUT)
    write(queue/'pipeline.json',dict(pid=os.getpid(),device=a.device,stage='complete' if code==0 else 'needs_attention',returncode=code))
    update_status(root)
    if code:raise SystemExit(code)


if __name__=='__main__':main()
