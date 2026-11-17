"""Train and evaluate the repaired source-entry composer on frozen task splits.

The default run uses five coupling levels and three seeds. --quick is an
interface smoke run, not evidence for the paper. Test access is a separate,
explicit command after checkpoint/code/protocol freezing.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import numpy as np
import torch
from .legacy_world import SFWorldExact
from .exact import canonical_sources, compose_fixed, evaluate_policy, evaluate_sources_sf, legacy_sources, optimal_policy
from .paper_aligned_lower import train_context_sf, load_context_sf
from .paper_aligned_run import train_upper, evaluate_attention, _torch_threads
from .paper_aligned_attention import PaperAlignedAttention
from .task_protocol import load_tasks, digest

HERE=Path(__file__).resolve().parents[1]
VERSION='ma-usfa-public-synthetic-v1'

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def new_json(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',encoding='utf8') as f:
        json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')

def sources_for(world):
    raw=legacy_sources(SFWorldExact(kappa=0,gamma=world.gamma,horizon=world.horizon))['policies']
    return canonical_sources(world,raw).copy()

def protocol_identity(protocol):
    meta=json.loads((protocol/'manifest.json').read_text())
    # No test objective parsing, or sealed-file open, is needed here.
    for split in ('train','validation'): load_tasks(protocol,split)
    return {'manifest':sha(protocol/'manifest.json'),
            'train':sha(protocol/meta['files']['train']['file']),
            'validation':sha(protocol/meta['files']['validation']['file']),
            'test_declared_sha256':meta['files']['test']['sha256']}

def _mode(name):
    return {'ddqn':'ddqn','pg':'pg','ppo-surrogate':'ppo'}[name]

def train(args):
    out=args.output.resolve()
    if out.exists() and any(out.iterdir()): raise FileExistsError('Output is nonempty; choose a fresh --output. Historical results are never overwritten.')
    out.mkdir(parents=True,exist_ok=True)
    protocol=args.protocol.resolve();identity=protocol_identity(protocol)
    train_tasks=load_tasks(protocol,'train'); val=load_tasks(protocol,'validation')
    if len(train_tasks)!=64 or len(val)!=16: raise ValueError('Expected the fixed 64/16 development objectives')
    if args.quick:
        kappas,seeds,budget,lower_steps=[1.],[100],640,640
    else:
        kappas,seeds,budget,lower_steps=args.kappas,args.seeds,args.budget,args.lower_steps
    if len(set(kappas))!=len(kappas) or len(set(seeds))!=len(seeds): raise ValueError('Duplicate kappa or seed')
    if min(budget,lower_steps)<=0 or budget%40 or lower_steps%40: raise ValueError('Budgets must be positive multiples of horizon 40')
    mode=_mode(args.optimizer); runs=[]; rows=[]
    new_json(out/'run_protocol.json',{'version':VERSION,'status':'RUNNING','optimizer':args.optimizer,
        'kappas':kappas,'seeds':seeds,'budget_upper':budget,'budget_lower':lower_steps,
        'objective_protocol':identity,'test_accessed':False,'quick_smoke_only':bool(args.quick),
        'selection':'one shared checkpoint maximizing all 16 validation objectives; earliest tie',
        'source_library':'four fixed kappa=0 source policies, never adapted on targets',
        'execution':'one own source entry per agent; execute its frozen primitive action',
        'ppo_semantics':'mode=ppo is a one-trajectory, one-update PPO-style PG surrogate; not multi-epoch PPO'})
    _torch_threads()
    for kappa in kappas:
        world=SFWorldExact(kappa=kappa); sources=sources_for(world)
        sf=evaluate_sources_sf(world,sources)
        for task in val:
            w=np.asarray(task['weights'],np.float32)
            fixed=compose_fixed(world,sources,sf,w)
            for m,p in fixed.items():rows.append({'kappa':kappa,'seed':None,'method':m,'task_id':task['id'],'kind':task['kind'],'return':evaluate_policy(world,p,w)['initial_return']})
            rows.append({'kappa':kappa,'seed':None,'method':'dp_optimal','task_id':task['id'],'kind':task['kind'],'return':optimal_policy(world,w)['initial_return']})
        for seed in seeds:
            run=out/f'kappa{kappa:g}_seed{seed}';run.mkdir()
            lower,_=train_context_sf(run/'lower',sources,train_tasks,seed=seed,steps=lower_steps,kappa=kappa)
            lower.eval()
            for p in lower.parameters():p.requires_grad_(False)
            model,summary=train_upper(run,world,sources,lower,train_tasks,val,seed=seed,budget=budget,mode=mode)
            if summary['steps']!=budget or summary['updates']<=0: raise RuntimeError('Incomplete upper training')
            result=evaluate_attention(model,world,val,sources,lower)
            for row in result['rows']:rows.append({'kappa':kappa,'seed':seed,'method':args.optimizer,**row})
            binding={'kappa':float(kappa),'seed':int(seed),'mode':mode,
                'budget_upper':int(budget),'budget_lower':int(lower_steps),
                'upper_sha256':sha(run/'upper.pt'),'lower_sha256':sha(run/'lower/lower.pt'),
                'source_array_sha256':hashlib.sha256(np.asarray(sources).tobytes()).hexdigest(),
                'objective_protocol':identity}
            new_json(run/'unit_binding.json',binding)
            runs.append({'kappa':float(kappa),'seed':int(seed),'mode':mode,'upper':str((run/'upper.pt').relative_to(out)),
                'lower':str((run/'lower/lower.pt').relative_to(out)),
                'binding':str((run/'unit_binding.json').relative_to(out)),
                'budget_upper':int(budget),'budget_lower':int(lower_steps)})
            print(json.dumps({'status':'UNIT_COMPLETE','kappa':kappa,'seed':seed,'validation':result['mean_return']}),flush=True)
    new_json(out/'completed.json',{'version':VERSION,'status':'COMPLETE','runs':runs,'objective_protocol':identity,
        'quick_smoke_only':bool(args.quick),'optimizer':args.optimizer,'test_accessed':False})
    new_json(out/'evaluation_validation.json',{'version':VERSION,'status':'PASS','split':'validation','rows':rows,
        'optimizer_updates':0,'test_accessed':False,'aggregation':'average objectives within each run; then independent training seeds',
        'quick_smoke_only':bool(args.quick)})
    print(json.dumps({'status':'COMPLETE','output':str(out),'test_accessed':False}),flush=True)

def freeze(args):
    root=args.output.resolve();complete=json.loads((root/'completed.json').read_text())
    if complete.get('status')!='COMPLETE':raise ValueError('Only completed development runs can be frozen')
    if complete.get('quick_smoke_only'):raise ValueError('Quick smoke runs cannot open the independent test')
    protocol=args.protocol.resolve()
    if protocol_identity(protocol)!=complete['objective_protocol']:raise ValueError('Objective protocol changed')
    files={str(p.relative_to(root)):sha(p) for p in [root/'completed.json',root/'run_protocol.json',root/'evaluation_validation.json']}
    if not complete.get('runs'):raise ValueError('No completed run to freeze')
    for r in complete['runs']:
        for key in ('upper','lower'):files[r[key]]=sha(root/r[key])
        if not r.get('binding'):raise ValueError('Public run lacks its upper/lower/world binding receipt')
        binding_path=root/r['binding'];binding=json.loads(binding_path.read_text())
        for key in ('kappa','seed','mode','budget_upper','budget_lower'):
            if binding.get(key)!=r[key]:raise ValueError('Unit binding differs from declared run')
        if (binding.get('upper_sha256')!=sha(root/r['upper'])
            or binding.get('lower_sha256')!=sha(root/r['lower'])
            or binding.get('objective_protocol')!=complete['objective_protocol']):
            raise ValueError('Upper/lower checkpoint pair does not match unit binding')
        files[r['binding']]=sha(binding_path)
        upper=torch.load(root/r['upper'],map_location='cpu',weights_only=False)
        lower=torch.load(root/r['lower'],map_location='cpu',weights_only=False)
        if (binding.get('source_array_sha256')!=upper.get('sources_sha256')
            or binding.get('source_array_sha256')!=lower.get('source_sha256')):
            raise ValueError('Unit binding source library differs from checkpoint identities')
        if (upper.get('seed')!=r['seed'] or upper.get('budget')!=r['budget_upper']
            or upper.get('mode')!=r['mode']):raise ValueError('Upper run identity mismatch')
        if (lower.get('seed')!=r['seed'] or lower.get('steps')!=r['budget_lower']
            or lower.get('kappa')!=r['kappa'] or lower.get('gamma')!=.95
            or lower.get('updates',0)<=0):raise ValueError('Lower run identity/budget mismatch')
        summary_path=(root/r['upper']).with_name('summary.json')
        summary=json.loads(summary_path.read_text())
        if summary.get('steps')!=r['budget_upper'] or summary.get('updates',0)<=0:raise ValueError('Upper run is partial')
        files[str(summary_path.relative_to(root))]=sha(summary_path)

    codes={p.name:sha(p) for p in Path(__file__).parent.glob('*.py')}
    doc={'version':VERSION,'status':'FROZEN','runs':complete['runs'],'optimizer':complete['optimizer'],
        'objective_protocol':complete['objective_protocol'],'files':files,'code':codes,'test_accessed':False,
        'test_semantics':'evaluation only; no model or checkpoint adaptation after first test access'}
    doc['sha256']=digest(doc);new_json(root/'frozen.json',doc)
    print(json.dumps({'status':'FROZEN','test_accessed':False}))

def evaluate_test(args):
    if not args.allow_test:raise PermissionError('Pass --allow-test only after finishing development and freezing models')
    root=args.output.resolve();doc=json.loads((root/'frozen.json').read_text());copy=dict(doc);expected=copy.pop('sha256')
    if digest(copy)!=expected:raise ValueError('Frozen manifest digest mismatch')
    for rel,h in doc['files'].items():
        if sha(root/rel)!=h:raise ValueError(f'Frozen artifact changed: {rel}')
    for rel,h in doc['code'].items():
        if sha(Path(__file__).parent/rel)!=h:raise ValueError(f'Frozen code changed: {rel}')
    protocol=args.protocol.resolve()
    if protocol_identity(protocol)!=doc['objective_protocol']:raise ValueError('Protocol changed')
    new_json(root/'test_access_started.json',{'status':'STARTED','frozen_sha256':expected,'unix_time':time.time(),
        'note':'Exclusive receipt prevents automatic retry, even after failure'})
    tasks=load_tasks(protocol,'test',allow_test=True);rows=[];worlds={};libs={};sf_cache={}
    _torch_threads()
    for r in doc['runs']:
        k=float(r['kappa'])
        if k not in worlds:
            worlds[k]=SFWorldExact(kappa=k);libs[k]=sources_for(worlds[k]);sf_cache[k]=evaluate_sources_sf(worlds[k],libs[k])
            for task in tasks:
                w=np.asarray(task['weights'],np.float32)
                for m,p in compose_fixed(worlds[k],libs[k],sf_cache[k],w).items():rows.append({'kappa':k,'seed':None,'method':m,'task_id':task['id'],'kind':task['kind'],'return':evaluate_policy(worlds[k],p,w)['initial_return']})
                rows.append({'kappa':k,'seed':None,'method':'dp_optimal','task_id':task['id'],'kind':task['kind'],'return':optimal_policy(worlds[k],w)['initial_return']})
        world,sources=worlds[k],libs[k];lower,_=load_context_sf(root/r['lower'])
        payload=torch.load(root/r['upper'],map_location='cpu',weights_only=False)
        if payload['mode']!=r['mode'] or payload['seed']!=r['seed'] or payload['budget']!=r['budget_upper']:raise ValueError('Upper identity mismatch')
        if payload['sources_sha256']!=hashlib.sha256(np.asarray(sources).tobytes()).hexdigest():raise ValueError('Source identity mismatch')
        model=PaperAlignedAttention();model.load_state_dict(payload['model']);model.eval();model.requires_grad_(False)
        before={n:p.detach().clone() for n,p in model.state_dict().items()}
        result=evaluate_attention(model,world,tasks,sources,lower)
        if any(not torch.equal(before[n],p) for n,p in model.state_dict().items()):raise RuntimeError('Evaluation changed model')
        for row in result['rows']:rows.append({'kappa':k,'seed':r['seed'],'method':doc['optimizer'],**row})
    bounds={(r['kappa'],r['task_id']):r['return'] for r in rows if r['method']=='dp_optimal'}
    if any(r['return']>bounds[(r['kappa'],r['task_id'])]+1e-6 for r in rows):raise RuntimeError('DP upper bound violation')
    new_json(root/'evaluation_test.json',{'version':VERSION,'status':'PASS','split':'test','rows':rows,
        'objective_count':len(tasks),'test_accessed':True,'optimizer_updates':0,'frozen_sha256':expected,
        'aggregation':'average objectives within each run; then independent training seeds'})
    print(json.dumps({'status':'PASS','split':'test','objective_count':len(tasks),'rows':len(rows),'optimizer_updates':0}))

def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage',choices=['train','freeze','test'],default='train')
    p.add_argument('--output',type=Path,default=HERE/'results/repaired')
    p.add_argument('--protocol',type=Path,default=HERE/'protocol')
    p.add_argument('--kappas',type=float,nargs='+',default=[0.,.25,.5,.75,1.])
    p.add_argument('--seeds',type=int,nargs='+',default=[100,101,102])
    p.add_argument('--budget',type=int,default=81920);p.add_argument('--lower-steps',type=int,default=81920)
    p.add_argument('--optimizer',choices=['ddqn','pg','ppo-surrogate'],default='ddqn')
    p.add_argument('--quick',action='store_true');p.add_argument('--allow-test',action='store_true')
    p.add_argument('--threads',type=int,choices=[1,2],default=2)
    p.add_argument('--max-rss-gib',type=float,default=3.);p.add_argument('--min-free-disk-gib',type=float,default=20.)
    p.add_argument('--wall-seconds',type=int,default=28800)
    p.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    return p

def main(argv=None):
    args=parser().parse_args(argv)
    os.environ['MA_USFA_THREAD_BUDGET']=str(args.threads)
    if not args.worker:
        from .resources import guarded_worker
        return guarded_worker(args)
    from .resources import WorkerGuard
    with WorkerGuard(args):
        {'train':train,'freeze':freeze,'test':evaluate_test}[args.stage](args)
    return 0

if __name__=='__main__':raise SystemExit(main())
