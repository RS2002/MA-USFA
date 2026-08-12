"""run_extend.py -- resumable driver for the EXTENDED toy experiments.

Builds on the verified N=2 crossover matrix (run_crossover_ckpt.py) and adds
three things the paper needs:

  * PER-SOURCE values V^k -- every training-set (library) strategy k is
    evaluated on the test task, and the Prop-1 check  L1 >= max_k V^k  is
    recorded in every cell, so "L1 is never worse than any source in the
    training set" is verified empirically (not just proved).

  * A SCALING SWEEP over the number of agents N = 3, 4, 5 on the A_distinct
    (null control) and B_overlap (crossover) tasks, checking that the L1<->L2
    crossover survives more agents and that L2's paid-region collapse
    sharpens with contention. joint_GPI is computed for N<=3 and OMITTED for
    N>=4 -- 5^4 = 625 / 5^5 = 3125 joint-action tuples make the A^N wall
    visible inside the toy itself.

  * A LIBRARY-SIZE SWEEP K = 1..8 on B_overlap, measuring the CCS-coverage
    effect: L1's free-region ceiling needs the contested-goal JOINT vertex
    (0,0) in the library (before coverage L1 sits below the joint optimum;
    at/after coverage it closes the gap), while the SAME coverage is a
    liability in the paid region because L1's per-state selection is
    collision-blind (harvest-only psi) -- coverage is regime-dependent, and
    only a learned composer (L2') is immune to library content.

Same schedule structure as run_crossover_ckpt.py so it fits short bash
budgets: 'prep' caches the library + the w-free USFA psi once per config;
'cell' computes one (task, kappa) base row (L1/L2/joint-GPI/per-source V^k +
collisions + retrain); 'comp' trains ONE composer flavor for a cell.

Usage:
  python run_extend.py prep  --N 2 --K 4 --lib corners
  python run_extend.py cell  --N 2 --K 4 --lib corners --task B_overlap --kappa 0.5
  python run_extend.py comp  --N 2 --K 4 --lib corners --task B_overlap --kappa 0.5 --flavor comm
  python run_extend.py tables [--mode main|sweep_N|sweep_K]
  python run_extend.py status [--mode main|sweep_N|sweep_K]
"""
import argparse
import json
import os
import numpy as np

from sfworld import SFWorld
from library import (build_library, make_L1_policy, make_L2_policy,
                     make_joint_gpi_policy, _states)
from sf_learn import train_q, make_q_policy, train_sf_usfa
from composers import Composer

GAMMA = 0.95
RES = 'results/ext.json'
EXT_DIR = 'results/ext'

USFA_EPS = {2: 8000, 3: 6000, 4: 5000, 5: 4000}
KAPPAS_MAIN = [0.0, 0.25, 0.5, 0.75, 1.0]
KAPPAS_SWEEP = [0.0, 0.5, 1.0]
FLAVORS_MAIN = ['qmix', 'attn', 'comm']
EVAL_N = 120
COMP_EPS = {2: 1000, 3: 2000, 4: 1600, 5: 1400}
RT_EPS = 4000


# --------------------------------------------------------------------- #
# config helpers
# --------------------------------------------------------------------- #
def n_features_for(N):
    return 5 if N == 5 else 4


def w_from_assign(assign, Kf, lo=0.15, dom=1.3):
    N = len(assign)
    w = np.full((N, Kf), lo)
    for i, c in enumerate(assign):
        w[i, c] = dom
    return w


def sources_for(N, K_lib, lib):
    """Source-task libraries. Every source is a corner/feature assignment
    (agent i -> feature assign[i]); w_from_assign turns it into [N,Kf]."""
    if lib == 'corners':
        pools = {
            2: [(0, 3), (3, 0), (1, 2), (2, 1)],
            3: [(0, 3, 1), (3, 0, 2), (1, 2, 0), (2, 1, 3)],
            4: [(0, 3, 1, 2), (3, 0, 2, 1), (1, 2, 0, 3), (2, 1, 3, 0)],
            # N=5 needs MORE sources so EVERY agent gets at least one source
            # whose own goal is the shared feature 0 (otherwise L2's free-region
            # gap would be a pure library-coverage artifact, not a method fact).
            5: [(0, 3, 1, 2, 4), (3, 0, 2, 1, 4), (1, 2, 0, 3, 4),
                (2, 1, 3, 0, 4), (1, 2, 3, 4, 0), (2, 3, 4, 1, 0),
                (3, 4, 2, 1, 0), (4, 3, 1, 2, 0)],
        }
        assert K_lib <= len(pools[N]), (N, K_lib)
        return pools[N][:K_lib]
    if lib == 'k_sweep':
        assert N == 2, "k_sweep is defined for N=2 (B_overlap)"
        libs = {
            1: [(0, 0)],
            2: [(0, 0), (1, 2)],
            3: [(0, 0), (1, 2), (2, 1)],
            4: [(0, 3), (3, 0), (1, 2), (2, 1)],
            5: [(0, 3), (3, 0), (1, 2), (2, 1), (0, 0)],
            6: [(0, 3), (3, 0), (1, 2), (2, 1), (0, 0), (1, 1)],
            8: [(0, 3), (3, 0), (1, 2), (2, 1), (0, 0), (1, 1), (2, 2),
                (3, 3)],
        }
        return libs[K_lib]
    raise ValueError(lib)


# weak secondary-goal index per agent (must match the verified N=2 numbers:
# agent0 -> corner 2, agent1 -> corner 3)
_SECONDARIES = {2: [2, 3], 3: [2, 3, 1], 4: [2, 3, 1, 2], 5: [2, 3, 1, 4, 2]}


def tasks_for(N, Kf):
    """Test tasks. A_distinct: agents to DISTINCT features (out-of-library;
    for N=2 the pair (0,1), which is NOT in the corner library). B_overlap:
    ALL agents dominant on feature 0 (shared goal) with weak per-agent
    secondaries -- the crossover task."""
    A = w_from_assign(list(range(N)), Kf)
    B = np.full((N, Kf), 0.15)
    B[:, 0] = 1.3
    for i, s in enumerate(_SECONDARIES[N]):
        B[i, s] = 0.8
    return {'A_distinct': A, 'B_overlap': B}


def make_env(N):
    return SFWorld(grid=5, n_agents=N, n_features=n_features_for(N),
                   resource_scale=0.6, horizon=40, seed=0,
                   collision_penalty=2.0)


def cfgkey(N, K_lib, lib):
    return f"N{N}_K{K_lib}_{lib}"


def cellkey(N, K_lib, lib, task, kappa):
    return f"{cfgkey(N, K_lib, lib)}|{task}|k{kappa}"


def evalp(env, pol, w, kappa, n=EVAL_N):
    return env.evaluate(pol, w, kappa, n_episodes=n, gamma=GAMMA)[0]


# --------------------------------------------------------------------- #
# per-source policies + Prop-1 check
# --------------------------------------------------------------------- #
def make_source_policy(item, env):
    """Replay source k's own greedy policy a^k_i = argmax psi^k_i . w^k_i."""
    psi, wsrc = item['psi'], item['w']
    N = env.n_agents

    def policy(positions, w):
        s = _states(env, positions)
        q = (psi[np.arange(N), s, :, :] * wsrc[:, None, :]).sum(-1)
        return q.argmax(-1)
    return policy


# --------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------- #
def load_results():
    if os.path.exists(RES):
        with open(RES) as f:
            return json.load(f)
    return {}


def save_results(r):
    os.makedirs('results', exist_ok=True)
    with open(RES, 'w') as f:
        json.dump(r, f, indent=1)


def stage_prep(N, K_lib, lib):
    env = make_env(N)
    Kf = n_features_for(N)
    assigns = sources_for(N, K_lib, lib)
    src_w = [w_from_assign(a, Kf) for a in assigns]
    libb = build_library(env, src_w, gamma=GAMMA)
    print(f"[prep] N={N} K={K_lib} lib={lib} built {len(libb)} sources")

    # w-free USFA psi is library-independent: train once per N, reuse.
    usfa_path = os.path.join(EXT_DIR, f"_usfa_N{N}.npz")
    if os.path.exists(usfa_path):
        psi = np.load(usfa_path)['usfa_psi']
        print(f"[prep] reusing usfa psi from {usfa_path}")
    else:
        print(f"[prep] training w-free USFA psi ({USFA_EPS[N]} ep, N={N}) ...")
        psi = train_sf_usfa(
            env,
            lambda rng: rng.uniform(0.1, 1.5, size=(N, Kf)),
            episodes=USFA_EPS[N],
            eps_decay=max(1500, USFA_EPS[N] // 2),
            gamma=GAMMA, seed=0)
        os.makedirs(EXT_DIR, exist_ok=True)
        np.savez_compressed(usfa_path, usfa_psi=psi)
        print(f"[prep] usfa psi cached -> {usfa_path}")

    os.makedirs(EXT_DIR, exist_ok=True)
    save = {'n_lib': np.array([len(libb)]), 'usfa_psi': psi}
    for i, item in enumerate(libb):
        save[f'lib_psi_{i}'] = item['psi']
        save[f'lib_w_{i}'] = item['w']
    cache = os.path.join(EXT_DIR, f"_c_{cfgkey(N, K_lib, lib)}.npz")
    np.savez_compressed(cache, **save)
    print(f"[prep] cached -> {cache}")


def load_cache(N, K_lib, lib):
    d = np.load(os.path.join(EXT_DIR, f"_c_{cfgkey(N, K_lib, lib)}.npz"))
    n = int(d['n_lib'][0])
    libb = [{'psi': d[f'lib_psi_{i}'], 'w': d[f'lib_w_{i}']} for i in range(n)]
    return libb, d['usfa_psi']


def _cells_for(N, K_lib, lib, mode, only=None):
    Kf = n_features_for(N)
    tasks = tasks_for(N, Kf)
    if mode == 'main':
        ks, tasks_sel, flavors = KAPPAS_MAIN, list(tasks), FLAVORS_MAIN
    elif mode == 'sweep_N':
        ks, tasks_sel, flavors = KAPPAS_SWEEP, list(tasks), ['comm']
    else:
        ks, tasks_sel, flavors = KAPPAS_SWEEP, ['B_overlap'], ['comm']
    out = []
    for t in tasks_sel:
        for k in ks:
            key = cellkey(N, K_lib, lib, t, k)
            if only and only != key:
                continue
            out.append((t, k, key, flavors))
    return out


def _row_done(row, flavors):
    return ('L1' in row and 'retrain' in row and
            all(f'L2prime_{fl}' in row for fl in flavors))


def stage_cell(N, K_lib, lib, task, kappa, mode):
    libb, _ = load_cache(N, K_lib, lib)
    env = make_env(N)
    L1 = make_L1_policy(libb, env)
    L2 = make_L2_policy(libb, env)
    JG = make_joint_gpi_policy(libb, env) if N <= 3 else None
    T = tasks_for(N, n_features_for(N))
    w = T[task]
    key = cellkey(N, K_lib, lib, task, kappa)
    have = load_results()
    row = have.get(key, {})
    if 'L1' not in row:
        row['L1'] = evalp(env, L1, w, kappa)
        row['L2'] = evalp(env, L2, w, kappa)
        row['joint_GPI'] = (evalp(env, JG, w, kappa) if JG is not None
                            else None)
        row['coll_L2'] = env.measure_conflicts(L2, w, kappa)
        row['coll_L1'] = env.measure_conflicts(L1, w, kappa)
        # per-source values V^k + Prop-1 check
        vs = [evalp(env, make_source_policy(item, env), w, kappa)
              for item in libb]
        row['Vk'] = vs
        row['max_Vk'] = float(max(vs))
        row['L1_minus_maxVk'] = float(row['L1'] - max(vs))
        have[key] = row
        save_results(have)
        print(f"[cell] {key}: L1={row['L1']:.2f} L2={row['L2']:.2f} "
              f"JG={row['joint_GPI'] if row['joint_GPI'] is None else round(row['joint_GPI'],2)} "
              f"collL2={row['coll_L2']:.3f} maxVk={row['max_Vk']:.2f} "
              f"L1-maxVk={row['L1_minus_maxVk']:+.2f}")
    # retrain (train_q on the true reward) once
    if 'retrain' not in row:
        Q = train_q(env, w, kappa=kappa, episodes=RT_EPS, gamma=GAMMA, seed=100)
        row['retrain'] = evalp(env, make_q_policy(Q), w, kappa)
        have[key] = row
        save_results(have)
        print(f"[cell] {key} retrain={row['retrain']:.2f}")


def stage_comp(N, K_lib, lib, task, kappa, flavor, mode):
    libb, psi = load_cache(N, K_lib, lib)
    env = make_env(N)
    T = tasks_for(N, n_features_for(N))
    w = T[task]
    key = cellkey(N, K_lib, lib, task, kappa)
    have = load_results()
    row = have.get(key, {})
    tag = f'L2prime_{flavor}'
    if tag in row:
        print(f"[comp] {key} {tag} already present")
        return
    comp = Composer(env, psi, w, flavor=flavor, gamma=GAMMA, seed=7)
    comp.train_task(w, kappa, episodes=COMP_EPS[N], seed=7)
    row[tag] = evalp(env, comp.policy(), w, kappa)
    have[key] = row
    save_results(have)
    print(f"[comp] {key} {tag}={row[tag]:.2f}")


def stage_status(mode):
    if mode == 'main':
        Ns, libs, Ks = [2], ['corners'], [4]
    elif mode == 'sweep_N':
        Ns, libs, Ks = [3, 4, 5], ['corners'], [4]
    else:
        Ns, libs, Ks = [2], ['k_sweep'], [1, 2, 3, 4, 5, 6, 8]
    have = load_results()
    for N in Ns:
        for lib in libs:
            for K in Ks:
                for (t, k, key, flavors) in _cells_for(N, K, lib, mode):
                    row = have.get(key, {})
                    if not _row_done(row, flavors):
                        print("pending:", key)


def stage_tables(mode):
    have = load_results()
    lines = []
    if mode == 'main':
        N, K_lib, lib = 2, 4, 'corners'
        lines.append("=== MAIN MATRIX (N=2, 4-source corner library) ===")
        lines.append("%-12s %7s %7s %7s %8s %8s %8s %7s %9s %9s" % (
            'task/kappa', 'L1', 'L2', 'jGPI', 'L2prime', 'retrain',
            'collL2', 'maxVk', 'L1-maxVk', 'order'))
        for t in tasks_for(N, n_features_for(N)):
            for k in KAPPAS_MAIN:
                r = have[cellkey(N, K_lib, lib, t, k)]
                bc = max(r[f'L2prime_{fl}']
                         for fl in FLAVORS_MAIN if f'L2prime_{fl}' in r)
                order = 'L2>L1' if r['L2'] > r['L1'] else 'L1>=L2'
                lines.append("%-12s %7.2f %7.2f %7.2f %8.2f %8.2f %8.3f "
                             "%7.2f %+9.2f %9s" % (
                    f'{t}|{k}', r['L1'], r['L2'],
                    (r['joint_GPI'] if r['joint_GPI'] is not None else float('nan')),
                    bc, r['retrain'], r['coll_L2'], r['max_Vk'],
                    r['L1_minus_maxVk'], order))
    elif mode == 'sweep_N':
        lines.append("=== AGENT-COUNT SWEEP (corner library, K=4) ===")
        lines.append("%-3s %-9s %7s %7s %7s %8s %8s %8s %9s" % (
            'N', 'task|kappa', 'L1', 'L2', 'jGPI', 'L2prime', 'retrain',
            'collL2', 'maxVk'))
        for N in [2, 3, 4, 5]:
            Kcell = 8 if N == 5 else 4
            for t in tasks_for(N, n_features_for(N)):
                for k in KAPPAS_SWEEP:
                    key = cellkey(N, Kcell, 'corners', t, k)
                    r = have[key]
                    jg = (r['joint_GPI'] if r['joint_GPI'] is not None
                          else float('nan'))
                    bc = max(r[f'L2prime_{fl}']
                             for fl in ('qmix', 'comm') if f'L2prime_{fl}' in r)
                    lines.append("%-3s %-9s %7.2f %7.2f %7.2f %8.2f %8.2f "
                                 "%8.3f %9.2f" % (
                        N, f'{t}|{k}', r['L1'], r['L2'], jg, bc,
                        r['retrain'], r['coll_L2'], r['max_Vk']))
    else:
        lines.append("=== LIBRARY-SIZE SWEEP (N=2, B_overlap) ===")
        lines.append("%-14s %-6s %7s %7s %8s %8s %8s %7s %9s" % (
            'library', 'kappa', 'L1', 'L2', 'L2prime', 'retrain', 'collL2',
            'maxVk', 'L1-maxVk'))
        labels = {
            1: '(0,0) only',
            2: '(0,0)+(1,2)',
            3: '(0,0)+(1,2)+(2,1)',
            4: 'P4 corners',
            5: 'P4+(0,0)',
            6: 'P4+(0,0)+(1,1)',
            8: 'P4+4 same',
        }
        for K in [1, 2, 3, 4, 5, 6, 8]:
            for k in KAPPAS_SWEEP:
                r = have[cellkey(2, K, 'k_sweep', 'B_overlap', k)]
                bc = max(r.get(f'L2prime_{fl}', float('-inf'))
                         for fl in ('qmix', 'comm'))
                lines.append("%-14s %-6s %7.2f %7.2f %8.2f %8.2f %8.3f "
                             "%7.2f %+9.2f" % (
                    labels[K], k, r['L1'], r['L2'], bc, r['retrain'],
                    r['coll_L2'], r['max_Vk'], r['L1_minus_maxVk']))
    txt = "\n".join(lines)
    with open('results/ext_tables.txt', 'w') as f:
        f.write(txt + "\n")
    print(txt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('stage', choices=['prep', 'cell', 'comp', 'tables',
                                      'status'])
    ap.add_argument('--N', type=int, default=2)
    ap.add_argument('--K', type=int, default=4)
    ap.add_argument('--lib', default='corners')
    ap.add_argument('--task', default='B_overlap')
    ap.add_argument('--kappa', type=float, default=0.5)
    ap.add_argument('--flavor', default='comm')
    ap.add_argument('--mode', default='main',
                    choices=['main', 'sweep_N', 'sweep_K'])
    ap.add_argument('--only', default=None)
    args = ap.parse_args()

    if args.stage == 'prep':
        stage_prep(args.N, args.K, args.lib)
    elif args.stage == 'cell':
        stage_cell(args.N, args.K, args.lib, args.task, args.kappa, args.mode)
    elif args.stage == 'comp':
        stage_comp(args.N, args.K, args.lib, args.task, args.kappa,
                   args.flavor, args.mode)
    elif args.stage == 'tables':
        stage_tables(args.mode)
    elif args.stage == 'status':
        stage_status(args.mode)


if __name__ == '__main__':
    main()
