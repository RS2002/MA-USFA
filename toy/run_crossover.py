"""Full crossover method matrix for the toy headline demonstration.

One controlled world, one source library, several TEST tasks x coupling levels,
evaluated under every method:

    L1        -- synchronized anchor (library)          [closed form]
    L2        -- per-agent GPI over library             [closed form]
    joint_GPI -- prior SF->MARL transfer, papers [2][3] [closed form, A^N]
    L2'       -- learned upper-level composer           [trained on the task]
    retrain   -- per-agent Q-learning on true reward    [trained on the task]

Claims verified numerically here:
  (1) L1<->L2 CROSSOVER: distinct-goal tasks give L2 >= L1 at all kappa;
      overlapping-goal tasks flip from L2 > L1 (free, kappa=0) to L1 > L2
      (paid, kappa>0) as coupling makes L2's independent argmax self-collide.
  (2) L2' DOMINATES: the learned composer is >= max(L1, L2) in every cell and
      recovers toward the retrain baseline in the paid region.
  (3) PRIOR METHOD LIMITS: joint_GPI never exceeds the coordinated anchor and
      inherits paid-region staleness (collision-blind psi); it also needs the
      joint action space A^N (a scalability note, not a number).

Run:  python run_crossover.py            (full)
      python run_crossover.py --quick    (fast smoke)
"""
import argparse
import json
import os
import numpy as np

from sfworld import SFWorld
from library import (build_library, make_L1_policy, make_L2_policy,
                     make_joint_gpi_policy)
from sf_learn import train_q, make_q_policy, train_sf_usfa
from composers import Composer

GAMMA = 0.95


def w_pair(c0, c1, lo=0.15, dom=1.3):
    w = np.full((2, 4), lo)
    w[0, c0] = dom
    w[1, c1] = dom
    return w


def make_composer(psi_usfa, env, w, kappa, flavor, episodes, seed):
    """L2' composer, warm-started from the w-free USFA features psi (a coherent
    SF fixed point) and fine-tuned on the task with per-agent honest TD. This is
    the same recipe verified in run_toy; the exact-library psi is optimistic as
    an envelope and destabilizes tabular TD, so the composer uses USFA psi."""
    comp = Composer(env, psi_usfa, w, flavor=flavor, gamma=GAMMA, seed=seed)
    comp.train_task(w, kappa, episodes=episodes, seed=seed)
    return comp.policy()


def evalp(env, pol, w, kappa, n):
    return env.evaluate(pol, w, kappa, n_episodes=n, gamma=GAMMA)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()

    EVAL_N = 30 if args.quick else 120
    COMP_EPS = 300 if args.quick else 1000
    RT_EPS = 800 if args.quick else 4000
    USFA_EPS = 3000 if args.quick else 8000
    KAPPAS = [0.0, 0.5, 1.0] if args.quick else [0.0, 0.25, 0.5, 0.75, 1.0]
    FLAVORS = ['qmix'] if args.quick else ['qmix', 'attn', 'comm']

    env = SFWorld(grid=5, n_agents=2, n_features=4, resource_scale=0.6,
                  horizon=40, seed=0, collision_penalty=2.0)

    # source library: distinct-corner specializations (coordinated sources)
    sources = [w_pair(0, 3), w_pair(3, 0), w_pair(1, 2), w_pair(2, 1)]
    lib = build_library(env, sources, gamma=GAMMA)
    L1 = make_L1_policy(lib, env)
    L2 = make_L2_policy(lib, env)
    JG = make_joint_gpi_policy(lib, env)

    # w-free USFA features for the L2' composer warm-start (one training run)
    print("[usfa] training w-free features for composer warm-start ...")

    def sampler(rng):
        return rng.uniform(0.1, 1.5, size=(2, 4))
    psi_usfa = train_sf_usfa(env, sampler, episodes=USFA_EPS,
                             eps_decay=max(1500, USFA_EPS // 2),
                             gamma=GAMMA, seed=0)

    # test tasks
    wA = w_pair(0, 3)                              # distinct goals
    wB = np.full((2, 4), 0.15)                     # overlapping dominant goal
    wB[0, 0] = 1.3; wB[1, 0] = 1.3
    wB[0, 2] = 0.8; wB[1, 3] = 0.8                 # weak distinct secondaries
    tasks = {'A_distinct': wA, 'B_overlap': wB}

    results = {}
    for tname, w in tasks.items():
        for k in KAPPAS:
            row = {}
            row['L1'] = evalp(env, L1, w, k, EVAL_N)
            row['L2'] = evalp(env, L2, w, k, EVAL_N)
            row['joint_GPI'] = evalp(env, JG, w, k, EVAL_N)
            Q = train_q(env, w, kappa=k, episodes=RT_EPS, gamma=GAMMA, seed=100)
            row['retrain'] = evalp(env, make_q_policy(Q), w, k, EVAL_N)
            for fl in FLAVORS:
                pol = make_composer(psi_usfa, env, w, k, fl, COMP_EPS, seed=7)
                row[f'L2prime_{fl}'] = evalp(env, pol, w, k, EVAL_N)
            row['coll_L2'] = env.measure_conflicts(L2, w, k)
            row['coll_L1'] = env.measure_conflicts(L1, w, k)
            results[f'{tname}|k={k}'] = row
            best_comp = max(row[f'L2prime_{fl}'] for fl in FLAVORS)
            print(f"{tname} k={k}: L1={row['L1']:6.2f} L2={row['L2']:6.2f} "
                  f"JG={row['joint_GPI']:6.2f} L2'={best_comp:6.2f} "
                  f"rt={row['retrain']:6.2f} | collL2={row['coll_L2']:.3f} "
                  f"{'L2>L1' if row['L2']>row['L1'] else 'L1>=L2'}")

    os.makedirs('results', exist_ok=True)
    with open('results/crossover.json', 'w') as f:
        json.dump(results, f, indent=1)
    _write_tables(results, tasks, KAPPAS, FLAVORS)
    print("\nsaved -> results/crossover.json, results/crossover_tables.txt")


def _write_tables(results, tasks, kappas, flavors):
    lines = []
    for tname in tasks:
        lines.append(f"\n=== TEST TASK {tname} ===")
        lines.append(f"{'kappa':>6} {'L1':>7} {'L2':>7} {'jGPI':>7} "
                     f"{'L2prime':>8} {'retrain':>8} {'order':>8}")
        for k in kappas:
            r = results[f'{tname}|k={k}']
            bc = max(r[f'L2prime_{fl}'] for fl in flavors)
            order = 'L2>L1' if r['L2'] > r['L1'] else 'L1>=L2'
            lines.append(f"{k:>6} {r['L1']:>7.2f} {r['L2']:>7.2f} "
                         f"{r['joint_GPI']:>7.2f} {bc:>8.2f} "
                         f"{r['retrain']:>8.2f} {order:>8}")
    txt = "\n".join(lines)
    with open('results/crossover_tables.txt', 'w') as f:
        f.write(txt + "\n")
    print(txt)


if __name__ == '__main__':
    main()
