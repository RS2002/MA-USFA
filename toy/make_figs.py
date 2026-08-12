"""make_figs.py -- renders of the three main-text toy figures.

Reads the cached results/ext.json (and, where present, results/crossover.json)
and writes the three figures back into results/.

Presentation rules (kept deliberately plain):
  * no in-figure titles (the LaTeX caption carries the description);
  * no in-figure text annotations or arrows (all analysis lives in the body);
  * reference lines and regions are explained through the legend or caption,
    never through text drawn inside the axes;
  * legend entries use the method names defined in the body
    (Synchronized / Independent / joint-GPI / MA-USFA / per-task retraining).

fig1_crossover.png : B_overlap (N=2, four-entry corner library): the kappa flip.
fig2_k_coverage.png: library-content sweep at N=2.
fig3_n_sweep.png   : team-size sweep, N=2..5.
"""
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

plt.rcParams['font.family'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['mathtext.fontset'] = 'dejavusans'

TOY = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(TOY, 'results')

with open(os.path.join(TOY, 'results', 'ext.json')) as f:
    R = json.load(f)


def cell(N, K, lib, task, k):
    return R[f'N{N}_K{K}_{lib}|{task}|k{k}']


def best_l2p(row):
    vals = [row[f'L2prime_{fl}'] for fl in ('qmix', 'attn', 'comm')
            if f'L2prime_{fl}' in row]
    return max(vals) if vals else np.nan


KAP = [0.0, 0.25, 0.5, 0.75, 1.0]
KAP3 = [0.0, 0.5, 1.0]
C = dict(l1='#c0392b', l2='#e67e22', jg='#7f8c8d', l2p='#1e8449', rt='#5b2c6f',
         b1='#2b5a8c')
XLAB = r'Transition coupling $\kappa$'
JOPT = 39.28

# ---------------------------------------------------------------- fig 1
fig, ax = plt.subplots(figsize=(7.0, 4.4))
B = {k: cell(2, 4, 'corners', 'B_overlap', k) for k in KAP}
ax.plot(KAP, [B[k]['L1'] for k in KAP], 'o-', color=C['l1'], lw=2,
        label='Synchronized')
ax.plot(KAP, [B[k]['L2'] for k in KAP], 'o-', color=C['l2'], lw=2,
        label='Independent')
ax.plot(KAP, [B[k]['joint_GPI'] for k in KAP], 's--', color=C['jg'], lw=1.5,
        label='joint-GPI')
ax.plot(KAP, [best_l2p(B[k]) for k in KAP], '^-', color=C['l2p'], lw=2,
        label='MA-USFA')
ax.plot(KAP, [B[k]['retrain'] for k in KAP], 'x--', color=C['rt'], lw=1.5,
        label='Per-task retraining')
ax.axhline(JOPT, color='#a0a0a0', ls=':', lw=1, label='Joint optimum')
ax.axvspan(-0.02, 0.02, color='#dfeefb', zorder=0)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0, 47)
ax.set_xticks(KAP)
ax.set_xlabel(XLAB)
ax.set_ylabel('Team return (sum over 40 steps)')
ax.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
ax.grid(alpha=0.25, ls=':')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig1_crossover.png'), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- fig 2
fig, ax = plt.subplots(figsize=(7.0, 4.4))
K4 = {k: cell(2, 4, 'k_sweep', 'B_overlap', k) for k in KAP}
K5 = {k: cell(2, 5, 'k_sweep', 'B_overlap', k) for k in KAP}
ax.plot(KAP, [K4[k]['L1'] for k in KAP], 'o-', color=C['b1'], lw=2,
        label='Synchronized, base library')
ax.plot(KAP, [K5[k]['L1'] for k in KAP], 'o-', color=C['l1'], lw=2,
        label='Synchronized, + contested entry')
ax.plot(KAP, [best_l2p(K5[k]) for k in KAP], '^-', color=C['l2p'], lw=2,
        label='MA-USFA')
ax.plot(KAP, [K5[k]['retrain'] for k in KAP], 'x--', color=C['rt'], lw=1.5,
        label='Per-task retraining')
ax.axhline(JOPT, color='#a0a0a0', ls=':', lw=1, label='Joint optimum')
ax.axvspan(-0.02, 0.02, color='#dfeefb', zorder=0)
ax.set_xlim(-0.02, 1.02)
ax.set_ylim(0, 47)
ax.set_xticks(KAP)
ax.set_xlabel(XLAB)
ax.set_ylabel('Team return (sum over 40 steps)')
ax.legend(loc='lower center', fontsize=8, framealpha=0.95, ncol=2)
ax.grid(alpha=0.25, ls=':')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig2_k_coverage.png'), dpi=200)
plt.close(fig)

# ---------------------------------------------------------------- fig 3
NS = [2, 3, 4, 5]
NC = {2: '#3498db', 3: '#e67e22', 4: '#8e44ad', 5: '#c0392b'}
fig, (axL, axR) = plt.subplots(1, 2, figsize=(11.0, 4.3))
for N in NS:
    Kcell = 8 if N == 5 else 4
    Bc = {k: cell(N, Kcell, 'corners', 'B_overlap', k) for k in KAP3}
    axL.plot(KAP3, [Bc[k]['L2'] for k in KAP3], 'o-', color=NC[N], lw=2,
             label=f'N={N}')
    axR.plot(KAP3, [best_l2p(Bc[k]) for k in KAP3], '^-', color=NC[N], lw=2,
             label=f'N={N}')
    axR.plot(KAP3, [Bc[k]['retrain'] for k in KAP3], 'x--', color=NC[N],
             lw=1.3, alpha=0.7)
axL.axhline(0, color='#a0a0a0', ls=':', lw=1)
axL.set_xlim(-0.02, 1.02)
axL.set_ylim(-45, 105)
axL.set_xticks(KAP3)
axL.set_xlabel(XLAB)
axL.set_ylabel('Team return (sum over 40 steps)')
axL.legend(title='Team size', fontsize=8.5)
axL.grid(alpha=0.25, ls=':')

axR.set_xlim(-0.02, 1.02)
axR.set_ylim(0, 105)
axR.set_xticks(KAP3)
axR.set_xlabel(XLAB)
axR.set_ylabel('Team return (sum over 40 steps)')
leg_N = axR.legend(title='Team size', fontsize=8.5, loc='upper left')
axR.add_artist(leg_N)
style_handles = [
    Line2D([0], [0], color='#444', ls='-', marker='^', label='MA-USFA'),
    Line2D([0], [0], color='#444', ls='--', marker='x',
           label='Per-task retraining'),
]
axR.legend(handles=style_handles, fontsize=8.5, loc='lower right')
axR.grid(alpha=0.25, ls=':')
fig.tight_layout()
fig.savefig(os.path.join(OUT, 'fig3_n_sweep.png'), dpi=200)
plt.close(fig)

print('wrote:')
for f in ('fig1_crossover.png', 'fig2_k_coverage.png', 'fig3_n_sweep.png'):
    p = os.path.join(OUT, f)
    print(' ', p, os.path.getsize(p), 'bytes')
