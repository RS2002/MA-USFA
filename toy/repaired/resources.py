"""Bound one owned CPU worker's memory high-water, disk and elapsed time."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time

def _rss_gib():
    if os.name=='nt':
        import psutil
        return psutil.Process().memory_info().rss/2**30
    import resource
    value=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value/(2**30 if sys.platform=='darwin' else 2**20)

class WorkerGuard:
    """Runs inside the sole owned worker; no training children are spawned."""
    def __init__(self,args):
        self.args=args;self.started=time.monotonic();self.stop=threading.Event();self.peak=0.;self.lowest=float('inf')
        self.disk_path=Path(args.output).resolve()
        while not self.disk_path.exists():self.disk_path=self.disk_path.parent
        self.check() # Fail before starting computation if measurement is unavailable.
    def snapshot(self):
        return {'peak_worker_rss_gib':self.peak,'lowest_free_disk_gib':self.lowest,
                'elapsed_seconds':time.monotonic()-self.started,'threads':self.args.threads}
    def check(self):
        rss=_rss_gib();free=shutil.disk_usage(self.disk_path).free/2**30
        self.peak=max(self.peak,rss);self.lowest=min(self.lowest,free)
        if rss>self.args.max_rss_gib:raise RuntimeError('worker memory high-water limit exceeded')
        if free<self.args.min_free_disk_gib:raise RuntimeError('free disk limit exceeded')
        if time.monotonic()-self.started>self.args.wall_seconds:raise RuntimeError('elapsed time limit exceeded')
    def _watch(self):
        while not self.stop.wait(1.):
            try:self.check()
            except BaseException as exc:
                print(json.dumps({'resource_guard':{'status':'STOPPED','reason':str(exc),**self.snapshot()}}),flush=True)
                # This is an isolated subprocess owned by the launcher. A
                # violation must stop numerical loops promptly, even inside BLAS.
                os._exit(75)
    def __enter__(self):
        self.thread=threading.Thread(target=self._watch,daemon=True);self.thread.start();return self
    def __exit__(self,*_):
        self.stop.set();self.thread.join(timeout=2)
        print(json.dumps({'resource_guard':{'status':'FINISHED',**self.snapshot()}}),flush=True)

def guarded_worker(args, entrypoint=None):
    if args.max_rss_gib<=0 or args.min_free_disk_gib<0 or args.wall_seconds<=0:raise ValueError('Invalid resource limits')
    env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','MA_USFA_THREAD_BUDGET'):
        env[key]=str(args.threads)
    argv=sys.argv[1:]+['--worker']
    worker=subprocess.Popen([sys.executable,str(entrypoint or (Path(__file__).resolve().parents[1]/'run_toy.py')),*argv],env=env)
    try:return worker.wait()
    except BaseException:
        worker.terminate()
        try:worker.wait(timeout=5)
        except subprocess.TimeoutExpired:worker.kill();worker.wait()
        raise
