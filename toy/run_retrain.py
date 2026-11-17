"""Validation-target-specific scratch DDQN diagnostic; never reads test weights.

This model is fitted separately on each requested validation target. Its
best-within-budget and endpoint values are development diagnostics, not
independent-test comparisons with a shared controller.
"""
import argparse
import json
import os
from pathlib import Path
import sys
from repaired.task_protocol import load_tasks, digest
from repaired.public_run import new_json, sha, protocol_identity
from repaired.scratch_learning import SweepConfig, train_dqn
from repaired.resources import WorkerGuard, guarded_worker

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--protocol',type=Path,default=Path(__file__).resolve().parent/'protocol')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--kappas',type=float,nargs='+',default=[0.,.25,.5,.75,1.])
    p.add_argument('--seed',type=int,default=100)
    p.add_argument('--task-id',nargs='+',default=['all'])
    p.add_argument('--budget',type=int,default=81920)
    p.add_argument('--threads',type=int,choices=[1,2],default=2)
    p.add_argument('--max-rss-gib',type=float,default=3.);p.add_argument('--min-free-disk-gib',type=float,default=20.)
    p.add_argument('--wall-seconds',type=int,default=28800)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    a=p.parse_args()
    if a.budget<=0 or a.budget%40:raise ValueError('budget must be a positive multiple of H=40')
    if len(set(a.kappas))!=len(a.kappas) or any(not 0<=k<=1 for k in a.kappas):raise ValueError('Invalid coupling levels')
    os.environ['MA_USFA_THREAD_BUDGET']=str(a.threads)
    if not a.worker:return guarded_worker(a,entrypoint=Path(__file__).resolve())
    with WorkerGuard(a):
        out=a.output.resolve()
        if out.exists() and any(out.iterdir()):raise FileExistsError('Fresh output directory required; no silent scratch restart')
        out.mkdir(parents=True,exist_ok=True);protocol=a.protocol.resolve()
        tasks=load_tasks(protocol,'validation')
        if a.task_id!=['all']:
            requested=set(a.task_id);tasks=[t for t in tasks if t['id'] in requested]
            if {t['id'] for t in tasks}!=requested:raise ValueError('Unknown validation target')
        identity={'version':'ma-usfa-public-validation-scratch-v1','budget':a.budget,'seed':a.seed,'kappas':a.kappas,
            'task_ids':[t['id'] for t in tasks],'objective_protocol':protocol_identity(protocol),'test_accessed':False,
            'selection':'endpoint and best exact return at or before fixed budget; same target as training',
            'architecture':'13->64->64->25, full Markov observation plus both task vectors',
            'learning_rate':.001,'code_sha256':sha(Path(__file__).parent/'repaired/scratch_learning.py')}
        new_json(out/'protocol.json',identity);pid=digest(identity);rows=[]
        for k in a.kappas:
            cfg=SweepConfig(kappa=k,direct_steps=a.budget,threads=a.threads,
                checkpoint_every=min(2560,a.budget),validation_every=min(2560,a.budget))
            for task in tasks:
                target=out/f'kappa{k:g}'/task['id']
                _,result=train_dqn(target,cfg,a.seed,'direct',a.budget,[task],[task],pid)
                if result.get('complete') is not True:raise RuntimeError('Scratch run interrupted; partial results preserved')
                progress=[json.loads(line) for line in (target/'progress.jsonl').read_text().splitlines() if line]
                vals=[r for r in progress if r.get('validation')]
                endpoint=next(r for r in reversed(vals) if r['steps']==a.budget)
                best=max(vals,key=lambda r:r['validation']['mean_return'])
                row={'kappa':k,'task_id':task['id'],'kind':task['kind'],'seed':a.seed,'budget':a.budget,
                    'endpoint_return':endpoint['validation']['mean_return'],'best_return':best['validation']['mean_return'],
                    'best_step':best['steps'],'steps':result['steps'],'updates':result['updates'],'test_accessed':False}
                new_json(target/'retrain_summary.json',row);rows.append(row)
        new_json(out/'summary.json',{'status':'PASS','version':identity['version'],'run_count':len(rows),'rows':rows,'test_accessed':False})
        print(json.dumps({'status':'PASS','run_count':len(rows),'test_accessed':False}),flush=True)
    return 0
if __name__=='__main__':raise SystemExit(main())
