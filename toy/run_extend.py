"""Scope/compatibility entry for the repaired synthetic release.

The audited current controlled protocol is N=2,K=4. Historical team/library
sweep numbers are not regenerated or silently relabeled as repaired evidence.
Use run_crossover.py for the verified five-coupling, three-seed run.
"""
import argparse,json
from pathlib import Path
from repaired.task_protocol import verify_protocol

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['status','prep','cell','comp','tables'],nargs='?',default='status')
    p.add_argument('--N',type=int,default=2);p.add_argument('--K',type=int,default=4)
    p.add_argument('--lib',default='corners');p.add_argument('--mode',default='main')
    p.add_argument('--task');p.add_argument('--kappa',type=float,default=1.);p.add_argument('--flavor',default='attn')
    a=p.parse_args()
    if a.N!=2 or a.K!=4 or a.lib!='corners':p.error('Current audited protocol supports N=2, K=4, fixed corner sources. New sweeps need a separately validated implementation.')
    if a.command in ['status','prep']:
        d=verify_protocol(Path(__file__).parent/'protocol')
        print(json.dumps({'status':'READY','N':2,'K':4,'counts':d['counts'],
            'run':'python run_crossover.py','quick':'python run_crossover.py --quick --output results/smoke',
            'historical_cached_results':'excluded from repaired evidence'}));return 0
    p.error('Per-target composer fitting and old sweep tables are retired. Use python run_crossover.py to train one shared controller per kappa/seed, or python make_figs.py --evaluation <current_evaluation.json>.')
if __name__=='__main__':raise SystemExit(main())
