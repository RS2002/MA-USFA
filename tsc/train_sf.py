"""SF+GPI training entry point for the TSC transfer experiment.

Three phases, selected by ``--phase``:

  * ``usfa``     -- train w-free successor features psi. A fresh task weight w is
                    drawn each episode (pushed onto the SF reward fn) so psi
                    learns the feature-based value without committing to a task.
                    Saves the trained psi to ``sf_psi.pth``.
  * ``transfer`` -- load a trained psi and EVALUATE the closed-form transfer
                    policy Q = psi . w on a target task (no learning). This is
                    the free-region method and the paid-region "before"
                    baseline.
  * ``composer`` -- load a trained psi, warm-start a NEIGHBOR-LIMITED composer
                    (1-hop road graph), and fine-tune it on a target task with
                    per-agent honest TD -- the paid-region recovery method.

The per-task retrain baseline is the existing IDDQN (``train.py``): a plain
per-agent DQN on the true task reward, trained from scratch at the coupled
regime. The comparison the paper reports is transfer (frozen psi) vs composer
(neighbour-limited fine-tune) vs retrain (IDDQN), analogous to the toy's
transfer / composer / train_q curves.

NOTE: heavy coupling in the TSC task is INTRINSIC (shared road cells, real
blocking) rather than a synthetic kappa dial -- there is no free knob to turn.
The transfer-vs-composer gap therefore measures how much the real network
coupling degrades closed-form SF transfer, and how much a neighbour-limited
composer recovers.
"""
import warnings
warnings.filterwarnings('ignore')
import os
os.environ["LIBSUMO_AS_TRACI"] = "1"

import argparse
import json
import numpy as np
import torch
import tqdm
import sumo_rl

from agent import SFAgent
from utils import parse_tripinfo, set_seed
from composite_reward import (make_sf_reward, DEFAULT_W, FEATURE_NAMES,
                              sample_task, sample_hetero_task,
                              make_source_library)

DATA_DIR = '../data/raw_data/manhattan_28x7'


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument('--phase', type=str, default='usfa',
                   choices=['usfa',
                            # paper names (preferred in the README / paper)
                            'independent', 'synchronized', 'ma-usfa',
                            # internal aliases (kept for backward compatibility)
                            'transfer', 'l1', 'composer'],
                   help="usfa: pre-train the shared w-free successor features "
                        "(prerequisite). The three composition rules on the "
                        "target task: 'independent' = per-agent greedy transfer; "
                        "'synchronized' = shared-anchor selection over a source "
                        "library; 'ma-usfa' = learned neighbour composer. "
                        "(transfer/l1/composer are accepted as legacy aliases.)")
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--gamma', type=float, default=0.99)
    p.add_argument('--max_step', type=int, default=60)
    p.add_argument('--episodes', type=int, default=500)
    p.add_argument('--eval_episode', type=int, default=5)
    p.add_argument('--eval_episodes', type=int, default=5,
                   help='rollouts averaged for transfer/composer evaluation. '
                        'Increase to e.g. 10 for multi-rollout mean/std reporting '
                        'without retraining (single-seed variance estimate).')
    p.add_argument('--cpu', action='store_true', default=False)
    p.add_argument('--cuda', type=str, default='0')
    p.add_argument('--hidden_dim', type=int, default=64)
    p.add_argument('--n_heads', type=int, default=4)
    p.add_argument('--use_agent_emb', type=int, default=1)
    p.add_argument('--buffer_size', type=int, default=100000)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--updates_per_step', type=int, default=1)
    p.add_argument('--target_update_freq', type=int, default=200)
    p.add_argument('--warmup', type=int, default=1000)
    p.add_argument('--eps_start', type=float, default=1.0)
    p.add_argument('--eps_end', type=float, default=0.05)
    p.add_argument('--eps_decay_episodes', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--sf_ckpt', type=str, default=None,
                   help='trained psi checkpoint (required for transfer/composer).')
    p.add_argument('--task', type=str, default=None,
                   help="target task w as comma-separated floats over "
                        f"{FEATURE_NAMES}; default = DEFAULT_W. Ignored in usfa.")
    p.add_argument('--task_seed', type=int, default=None,
                   help='if set, sample the target task from this seed instead '
                        'of --task (for a batch of random target tasks).')
    p.add_argument('--hetero', type=int, default=0,
                   help='1: w-HETEROGENEOUS target task -- draw an independent '
                        'per-intersection w (random per node, fixed DEFAULT_W '
                        'signs). 0: w-homogeneous (one shared w). Uses --task_seed '
                        '(or --seed) as the generator seed. Ignored in usfa.')
    p.add_argument('--lib_size', type=int, default=None,
                   help='Number of source profiles for the L1 anchor library '
                        '(K-sweep experiment). Default None = full library (5 profiles: '
                        '1 balanced + 4 specialists). K=1: balanced only; K=2: +queue; '
                        'K=3: +wait; K=4: +pressure; K=5: +speed. Ignored outside l1.')
    p.add_argument('--tripinfo', type=int, default=1)
    args = p.parse_args()
    # Normalize the paper-facing phase names to the internal tokens the rest of
    # this file switches on. The paper refers to the three composition rules as
    # Independent / Synchronized / MA-USFA; the code historically called them
    # transfer / l1 / composer.
    _phase_alias = {'independent': 'transfer',
                    'synchronized': 'l1',
                    'ma-usfa': 'composer'}
    args.phase = _phase_alias.get(args.phase, args.phase)
    return args


def parse_task(args, num_agents=None):
    """Target task for transfer/l1/composer.

    w-homogeneous -> a single [K] vector (from --task, --task_seed, or DEFAULT_W).
    w-heterogeneous (--hetero 1) -> a [num_agents, K] matrix, one independent w
    per intersection (random magnitudes, fixed signs)."""
    if args.hetero:
        assert num_agents is not None, "hetero task needs num_agents"
        seed = args.task_seed if args.task_seed is not None else args.seed
        return sample_hetero_task(np.random.RandomState(seed), num_agents)
    if args.task_seed is not None:
        return sample_task(np.random.RandomState(args.task_seed))
    if args.task:
        return np.array([float(x) for x in args.task.split(',')], dtype=np.float32)
    return DEFAULT_W.copy()


def build_env(reward_fn, args, tripinfo_path=None):
    delta_time = 10
    extra = "--scale 2.0 --no-warnings"
    if tripinfo_path:
        extra += f" --tripinfo-output {tripinfo_path}"
    return sumo_rl.parallel_env(
        net_file=f'{DATA_DIR}/manhattan_28x7.net.xml',
        route_file=f'{DATA_DIR}/manhattan_28x7.rou.xml',
        use_gui=False, num_seconds=args.max_step * delta_time, delta_time=delta_time,
        yellow_time=2, min_green=5, max_green=50,
        reward_fn=reward_fn, additional_sumo_cmd=extra)


def _apply_task(agent, reward_fn, w, args):
    """Push the target task onto BOTH the operator (agent.w) and the environment
    reward (reward_fn), handling homogeneous [K] and heterogeneous [N,K] w."""
    agent.w = w
    if args.hetero:
        # per-intersection reward weights, keyed by the agent's sorted ids
        w_map = {ts: w[i] for i, ts in enumerate(agent.id)}
        if hasattr(reward_fn, 'set_w_map'):
            reward_fn.set_w_map(w_map)
    else:
        if hasattr(reward_fn, 'set_w'):
            reward_fn.set_w(w)


def rollout_metrics(agent, args, tripinfo_path=None, deterministic=True):
    """One greedy rollout; returns (total phi.w return, full metric dict).

    The metric dict carries the physical TSC panel every method is compared on:
    avg_queue, avg_wait, avg_speed, fairness dispersion (from get_stats) and
    travel_time / throughput (from the SUMO tripinfo file)."""
    observe = agent.reset(train=False)
    total = 0.0
    for _ in range(args.max_step):
        action_dic = agent.act(observe, deterministic=deterministic)
        observe, reward, done = agent.step(action_dic)
        total += reward
        if done:
            break
    if tripinfo_path:
        try:
            agent.env.close()
            tm = parse_tripinfo(tripinfo_path)
            agent.travel_time = tm['travel_time']
            agent.throughput = tm['throughput']
        except Exception as e:
            print(f"[tripinfo] parse skipped: {e}")
    return total, agent.get_stats()


def main():
    args = get_args()
    set_seed(args.seed)
    device = torch.device("cuda:" + args.cuda
                          if torch.cuda.is_available() and not args.cpu else 'cpu')

    run_dir = os.path.abspath(f"./sf_s{args.seed}/{args.phase}")
    os.makedirs(run_dir, exist_ok=True)

    if args.phase == 'usfa':
        # w-free training: reward fn's w is overwritten each episode by the agent.
        reward_fn = make_sf_reward(DEFAULT_W)
        env = build_env(reward_fn, args)
        agent = SFAgent(env, reward_fn, device, mode='usfa', hidden_dim=args.hidden_dim,
                        lr=args.lr, gamma=args.gamma, horizon_len=args.max_step,
                        training_episodes=args.episodes, buffer_size=args.buffer_size,
                        n_heads=args.n_heads, use_agent_emb=bool(args.use_agent_emb),
                        target_update_freq=args.target_update_freq)
        agent.set_base_seed(args.seed)
        _train_loop(agent, args, run_dir, learn=True)
        agent.save_sf(os.path.join(run_dir, 'sf_psi.pth'))
        print(f"[usfa] saved psi -> {os.path.join(run_dir, 'sf_psi.pth')}")

    elif args.phase == 'transfer':
        assert args.sf_ckpt, "transfer requires --sf_ckpt (trained psi)"
        # need num_agents for hetero w -> build a throwaway to read agent ids,
        # but simpler: build env+agent first with DEFAULT_W, then apply the task.
        reward_fn = make_sf_reward(DEFAULT_W)
        trip = os.path.join(run_dir, 'tripinfo.xml') if args.tripinfo else None
        env = build_env(reward_fn, args, tripinfo_path=trip)
        agent = SFAgent(env, reward_fn, device, mode='usfa', hidden_dim=args.hidden_dim,
                        gamma=args.gamma, horizon_len=args.max_step,
                        use_agent_emb=bool(args.use_agent_emb), sf_ckpt=args.sf_ckpt)
        agent.set_base_seed(args.seed)
        w = parse_task(args, num_agents=agent.num)
        _apply_task(agent, reward_fn, w, args)
        results = [rollout_metrics(agent, args, tripinfo_path=trip)
                   for _ in range(args.eval_episodes)]
        _report(run_dir, 'transfer', w, results, args)

    elif args.phase == 'l1':
        assert args.sf_ckpt, "l1 requires --sf_ckpt (trained psi)"
        reward_fn = make_sf_reward(DEFAULT_W)
        trip = os.path.join(run_dir, 'tripinfo.xml') if args.tripinfo else None
        env = build_env(reward_fn, args, tripinfo_path=trip)
        src_lib = make_source_library(lib_size=args.lib_size)
        print(f"[l1] source library: {len(src_lib)} profiles "
              f"(lib_size={args.lib_size})")
        agent = SFAgent(env, reward_fn, device, mode='l1', hidden_dim=args.hidden_dim,
                        gamma=args.gamma, horizon_len=args.max_step,
                        use_agent_emb=bool(args.use_agent_emb), sf_ckpt=args.sf_ckpt,
                        source_library=src_lib)
        agent.set_base_seed(args.seed)
        w = parse_task(args, num_agents=agent.num)
        _apply_task(agent, reward_fn, w, args)
        results = [rollout_metrics(agent, args, tripinfo_path=trip)
                   for _ in range(args.eval_episodes)]
        _report(run_dir, 'l1', w, results, args)

    else:  # composer
        assert args.sf_ckpt, "composer requires --sf_ckpt (trained psi)"
        reward_fn = make_sf_reward(DEFAULT_W)
        trip = os.path.join(run_dir, 'tripinfo.xml') if args.tripinfo else None
        env = build_env(reward_fn, args, tripinfo_path=trip)
        agent = SFAgent(env, reward_fn, device, mode='composer', hidden_dim=args.hidden_dim,
                        lr=args.lr, gamma=args.gamma, horizon_len=args.max_step,
                        training_episodes=args.episodes, buffer_size=args.buffer_size,
                        n_heads=args.n_heads, use_agent_emb=bool(args.use_agent_emb),
                        target_update_freq=args.target_update_freq, sf_ckpt=args.sf_ckpt)
        agent.set_base_seed(args.seed)
        w = parse_task(args, num_agents=agent.num)
        _apply_task(agent, reward_fn, w, args)
        # --episodes 0: skip training and load the existing composer checkpoint.
        # This is used for multi-rollout variance estimation (EXP2) where the
        # trained composer.pth already exists and we only want repeated evals.
        composer_ckpt = os.path.join(run_dir, 'composer.pth')
        if args.episodes == 0:
            if os.path.exists(composer_ckpt):
                agent.load(composer_ckpt)
                print(f"[composer] --episodes 0: loaded existing checkpoint "
                      f"{composer_ckpt}, skipping training.")
            else:
                print(f"[composer] WARNING: --episodes 0 but no checkpoint found "
                      f"at {composer_ckpt}. Evaluating warm-started (untrained) composer.")
        else:
            _train_loop(agent, args, run_dir, learn=True, fixed_w=True)
            agent.save(composer_ckpt)
        results = [rollout_metrics(agent, args, tripinfo_path=trip)
                   for _ in range(args.eval_episodes)]
        _report(run_dir, 'composer', w, results, args)


def _train_loop(agent, args, run_dir, learn, fixed_w=False):
    log_path = os.path.join(run_dir, 'train.txt')
    global_env_steps = 0
    for epoch in range(1, args.episodes + 1):
        frac = min(1.0, (epoch - 1) / max(1, args.eps_decay_episodes))
        eps = args.eps_start + frac * (args.eps_end - args.eps_start)
        observe = agent.reset(train=True, episode=epoch)
        total, ep_losses = 0.0, []
        for t in range(args.max_step):
            step_eps = 1.0 if global_env_steps < args.warmup else eps
            action_dic = agent.act(observe, exploration_rate=step_eps)
            observe, reward, done = agent.step(action_dic)
            total += reward
            global_env_steps += 1
            if learn and global_env_steps >= args.warmup:
                l = agent.learn(batch_size=args.batch_size, updates=args.updates_per_step)
                if l:
                    ep_losses.append(l)
            if done:
                break
        agent.scheduler.step()
        loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        log = f"Epoch {epoch} | Reward {total:.2f} | Loss {loss:.4f} | Eps {eps:.3f}"
        print(log)
        with open(log_path, 'a') as f:
            f.write(log + "\n")


def _report(run_dir, phase, w, results, args):
    """Write the full physical-metric panel (not just reward) for one method.

    ``results`` is a list of (total_return, stats_dict) from rollout_metrics.
    Every method (retrain via train.py, L1, L2=transfer, L2'=composer) is
    reported on the SAME panel so the comparison is on real traffic metrics,
    not only the phi.w objective: reward, travel_time, throughput, avg_queue,
    avg_wait, avg_speed, fairness dispersion."""
    vals = [r for r, _ in results]
    stats = [s for _, s in results]

    def _mean(key):
        xs = [s.get(key, float('nan')) for s in stats]
        xs = [x for x in xs if x == x]                 # drop NaNs
        return float(np.mean(xs)) if xs else float('nan')

    panel = {k: _mean(k) for k in
             ('travel_time', 'throughput', 'avg_queue', 'avg_wait',
              'avg_speed', 'avg_fairness')}
    out = {
        'phase': phase,
        'hetero': bool(args.hetero),
        'w': (w.tolist() if hasattr(w, 'tolist') else [float(x) for x in w]),
        'mean_return': float(np.mean(vals)),
        'std_return': float(np.std(vals)),
        'n': len(vals),
        'returns': [float(v) for v in vals],
        'metrics': panel,
    }
    tag = 'hetero' if args.hetero else 'homo'
    with open(os.path.join(run_dir, f'{phase}_{tag}_result.json'), 'w') as f:
        json.dump(out, f, indent=1)
    print(f"[{phase}|{tag}] mean_return={out['mean_return']:.2f} +- {out['std_return']:.2f} "
          f"(n={out['n']}) | travel_time={panel['travel_time']:.2f} "
          f"throughput={panel['throughput']:.0f} queue={panel['avg_queue']:.2f} "
          f"wait={panel['avg_wait']:.2f} speed={panel['avg_speed']:.2f} "
          f"fairness={panel['avg_fairness']:.3f}")


if __name__ == '__main__':
    main()
