"""Plot current repaired evaluations; never reads historical cached table JSON."""
import argparse,json
from collections import defaultdict
from pathlib import Path
import numpy as np

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,default=Path(__file__).parent/'results/repaired/evaluation_validation.json')
    p.add_argument('--output',type=Path,default=Path(__file__).parent/'results/repaired/comparison.png')
    a=p.parse_args()
    if not a.evaluation.is_file():p.error('No repaired evaluation found. Run run_crossover.py first or pass --evaluation. Historical crossover.json/ext.json are deliberately excluded.')
    d=json.loads(a.evaluation.read_text())
    if d.get('status')!='PASS' or d.get('version')!='ma-usfa-public-synthetic-v1' or d.get('optimizer_updates')!=0:p.error('Expected a complete repaired evaluation-only report')
    groups=defaultdict(list)
    for row in d['rows']:groups[(row['method'],float(row['kappa']),row['seed'])].append(float(row['return']))
    by=defaultdict(list)
    for (method,kappa,seed),v in groups.items():by[(method,kappa)].append(float(np.mean(v)))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4.5))
    for method in sorted({m for m,k in by}):
        ks=sorted(k for m,k in by if m==method);means=[np.mean(by[(method,k)]) for k in ks]
        std=[np.std(by[(method,k)],ddof=1) if len(by[(method,k)])>1 else 0 for k in ks]
        ax.errorbar(ks,means,yerr=std,marker='o',capsize=3,label=method)
    ax.set(xlabel='Coupling kappa',ylabel='Discounted team return',title='Repaired '+d['split']+' evaluation')
    ax.legend(fontsize=8);fig.tight_layout();a.output.parent.mkdir(parents=True,exist_ok=True)
    if a.output.exists():raise FileExistsError('Plot output exists; choose a fresh --output')
    fig.savefig(a.output,dpi=180);plt.close(fig);print(a.output)
if __name__=='__main__':main()
