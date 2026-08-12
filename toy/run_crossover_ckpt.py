"""Resumable driver for the crossover matrix (fits short compute budgets).

Splits run_crossover into cacheable stages so it can be driven across many
short invocations:

  stage 'prep'  : build library + train the w-free USFA psi ONCE, cache to
                  results/_cache.npz  (library psi/w + usfa psi).
  stage 'cell'  : compute ONE (task,kappa) cell of the method matrix and append
                  it to results/crossover.json; skips cells already present.
  stage 'tables': render results/crossover_tables.txt from crossover.json.

Same numbers/logic as run_crossover.py; only the scheduling differs.

Usage:
  python run_crossover_ckpt.py prep   [--quick]
  python run_crossover_ckpt.py cell   [--quick] [--only tname:k]  (one pending cell)
  python run_crossover_ckpt.py tables [--quick]
  python run_crossover_ckpt.py status [--quick]
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
CACHE = 'results/_cache.npz'
RES = 'results/crossover.json'


def w_pair(c0, c1, lo=0.15, dom=1.3):
    w = np.full((2, 4), lo)
    w[0, c0] = dom
    w[1, c1] = dom
    return w


def cfg(quick):
    return dict(
        EVAL_N=30 if quick else 120,
        COMP_EPS=300 if quick else 1000,
        RT_EPS=800 if quick else 4000,
        USFA_EPS=3000 if quick else 8000,
        KAPPAS=[0.0, 0.5, 1.0] if quick else [0.0, 0.25, 0.5, 0.75, 1.0],
        FLAVORS=['qmix'] if quick else ['qmix', 'attn', 'comm'],
    )


def make_env():
    return SFWorld(grid=5, n_agents=2, n_features=4, resource_scale=0.6,
                   horizon=40, seed=0, collision_penalty=2.0)


def tasks():
    wA = w_pair(0, 3)
    wB = np.full((2, 4), 0.15)
    wB[0, 0] = 1.3; wB[1, 0] = 1.3
    wB[0, 2] = 0.8; wB[1, 3] = 0.8
    return {'A_distinct': wA, 'B_overlap': wB}


def sources():
    return [w_pair(0, 3), w_pair(3, 0), w_pair(1, 2), w_pair(2, 1)]


def evalp(env, pol, w, kappa, n):
    return env.evaluate(pol, w, kappa, n_episodes=n, gamma=GAMMA)[0]


def _load_results():
    if os.path.exists(RES):
        with open(RES) as f:
            return json.load(f)
    return {}


def _save_results(r):
    os.makedirs('results', exist_ok=True)
    with open(RES, 'w') as f:
        json.dump(r, f, indent=1)


def stage_prep(quick):
    c = cfg(quick)
    env = make_env()
    lib = build_library(env, sources(), gamma=GAMMA)
    print("[prep] library built:", len(lib), "sources")
    print("[prep] training w-free USFA psi (%d ep) ..." % c['USFA_EPS'])
    psi = train_sf_usfa(env, lambda rng: rng.uniform(0.1, 1.5, size=(2, 4)),
                        episodes=c['USFA_EPS'],
                        eps_decay=max(1500, c['USFA_EPS'] // 2),
                        gamma=GAMMA, seed=0)
    os.makedirs('results', exist_ok=True)
    save = {'usfa_psi': psi, 'n_lib': np.array([len(lib)])}
    for i, item in enumerate(lib):
        save['lib_psi_%d' % i] = item['psi']
        save['lib_w_%d' % i] = item['w']
    np.savez_compressed(CACHE, **save)
    print("[prep] cached ->", CACHE)


def _load_cache():
    d = np.load(CACHE)
    n = int(d['n_lib'][0])
    lib = [{'psi': d['lib_psi_%d' % i], 'w': d['lib_w_%d' % i]}
           for i in range(n)]
    return lib, d['usfa_psi']


def _pending(quick):
    c = cfg(quick)
    have = _load_results()
    todo = []
    for tname in tasks():
        for k in c['KAPPAS']:
            key = '%s|k=%s' % (tname, k)
            row = have.get(key, {})
            done = ('L1' in row and
                    all('L2prime_%s' % fl in row for fl in c['FLAVORS']))
            if not done:
                todo.append((tname, k, key))
    return todo


def stage_cell(quick, only=None):
    c = cfg(quick)
    lib, psi_usfa = _load_cache()
    env = make_env()
    L1 = make_L1_policy(lib, env)
    L2 = make_L2_policy(lib, env)
    JG = make_joint_gpi_policy(lib, env)
    T = tasks()

    # find first cell that is not fully complete (base + all flavors)
    have = _load_results()
    target = None
    for tname in tasks():
        for k in c['KAPPAS']:
            key = '%s|k=%s' % (tname, k)
            if only and key != only:
                continue
            row = have.get(key, {})
            done = ('L1' in row and
                    all('L2prime_%s' % fl in row for fl in c['FLAVORS']))
            if not done:
                target = (tname, k, key)
                break
        if target:
            break
    if target is None:
        print("[cell] nothing pending")
        return

    tname, k, key = target
    w = T[tname]
    row = have.get(key, {})

    # base (cheap ops + retrain) computed once
    if 'L1' not in row:
        row['L1'] = evalp(env, L1, w, k, c['EVAL_N'])
        row['L2'] = evalp(env, L2, w, k, c['EVAL_N'])
        row['joint_GPI'] = evalp(env, JG, w, k, c['EVAL_N'])
        Q = train_q(env, w, kappa=k, episodes=c['RT_EPS'], gamma=GAMMA, seed=100)
        row['retrain'] = evalp(env, make_q_policy(Q), w, k, c['EVAL_N'])
        row['coll_L2'] = env.measure_conflicts(L2, w, k)
        row['coll_L1'] = env.measure_conflicts(L1, w, k)
        have[key] = row
        _save_results(have)
        print("[cell] %s base: L1=%6.2f L2=%6.2f JG=%6.2f rt=%6.2f collL2=%.3f"
              % (key, row['L1'], row['L2'], row['joint_GPI'], row['retrain'],
                 row['coll_L2']))

    # one composer flavor per invocation
    for fl in c['FLAVORS']:
        tag = 'L2prime_%s' % fl
        if tag in row:
            continue
        comp = Composer(env, psi_usfa, w, flavor=fl, gamma=GAMMA, seed=7)
        comp.train_task(w, k, episodes=c['COMP_EPS'], seed=7)
        row[tag] = evalp(env, comp.policy(), w, k, c['EVAL_N'])
        have[key] = row
        _save_results(have)
        print("[cell] %s %s=%6.2f" % (key, tag, row[tag]))
        return

    print("[cell] %s complete" % key)


def stage_status(quick):
    todo = _pending(quick)
    have = _load_results()
    print("[status] done=%d pending=%d" % (len(have), len(todo)))
    for t in todo:
        print("   pending:", t[2])


def stage_tables(quick):
    c = cfg(quick)
    results = _load_results()
    lines = []
    for tname in tasks():
        lines.append("\n=== TEST TASK %s ===" % tname)
        lines.append("%6s %7s %7s %7s %8s %8s %8s" % (
            'kappa', 'L1', 'L2', 'jGPI', 'L2prime', 'retrain', 'order'))
        for k in c['KAPPAS']:
            r = results['%s|k=%s' % (tname, k)]
            bc = max(r['L2prime_%s' % fl] for fl in c['FLAVORS'])
            order = 'L2>L1' if r['L2'] > r['L1'] else 'L1>=L2'
            lines.append("%6s %7.2f %7.2f %7.2f %8.2f %8.2f %8s" % (
                k, r['L1'], r['L2'], r['joint_GPI'], bc, r['retrain'], order))
    txt = "\n".join(lines)
    with open('results/crossover_tables.txt', 'w') as f:
        f.write(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prep', 'cell', 'tables', 'status'])
    ap.add_argument('--quick', action='store_true')
    ap.add_argument('--only', default=None)
    args = ap.parse_args()
    if args.stage == 'prep':
        stage_prep(args.quick)
    elif args.stage == 'cell':
        stage_cell(args.quick, only=args.only)
    elif args.stage == 'tables':
        stage_tables(args.quick)
    elif args.stage == 'status':
        stage_status(args.quick)


if __name__ == '__main__':
    main()
