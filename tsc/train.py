import warnings
warnings.filterwarnings('ignore')
import os
os.environ["LIBSUMO_AS_TRACI"] = "1"

from agent import Agent
from utils import parse_tripinfo, set_seed
from composite_reward import make_composite_reward
import argparse
import tqdm
import torch
import numpy as np
import sumo_rl

# ==================== Reward ====================
# Same reward families as the other baselines so results are comparable:
#   (A) sumo_rl BUILT-IN single-metric rewards (queue/diffwait/pressure) for the
#       k-hop analysis of the CMAT/MAPO methods. IDDQN has no hop, but the reward
#       switch is kept identical for apples-to-apples runs.
#   (B) 'composite' -- the hand-crafted multi-term reward for the MAIN comparison
#       experiment (see composite_reward.py).
REWARD_MAP = {
    'diffwait': 'diff-waiting-time',
    'queue': 'queue',
    'pressure': 'pressure',
    'composite': 'composite',   # sentinel; resolved to a callable at env build
}


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--max_step', type=int, default=60)
    parser.add_argument('--eval_episode', type=int, default=5)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--cuda", type=str, default='0')
    parser.add_argument('--hidden_dim', type=int, default=64)
    # ---- IDDQN network: MLP (default, fully independent) or GAT (1-hop) ----
    parser.add_argument('--net', type=str, default='mlp', choices=['mlp', 'gat'],
                        help="Q-network. 'mlp' (default): fully INDEPENDENT "
                             "per-agent Q (no cross-agent flow). 'gat': one "
                             "1-hop graph-attention layer so each agent's Q also "
                             "depends on its immediate road neighbours.")
    parser.add_argument('--n_heads', type=int, default=4,
                        help='GAT attention heads (ignored for --net mlp).')
    parser.add_argument('--use_agent_emb', type=int, default=1,
                        help='1: per-intersection identity embedding (heterogeneous). '
                             '0: homogeneous agents.')
    # ---- Double DQN / target sync / replay ----
    parser.add_argument('--double', type=int, default=1,
                        help='1: Double DQN target (online selects, target evaluates). '
                             '0: vanilla DQN max target.')
    parser.add_argument('--buffer_size', type=int, default=100000,
                        help='experience-replay capacity (transitions).')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--updates_per_step', type=int, default=1,
                        help='DQN gradient steps taken per environment step.')
    parser.add_argument('--target_update_freq', type=int, default=200,
                        help='hard target-network sync every N gradient steps.')
    parser.add_argument('--warmup', type=int, default=1000,
                        help='env steps of random experience collected before '
                             'learning starts (fills the replay buffer).')
    # ---- epsilon-greedy schedule ----
    parser.add_argument('--eps_start', type=float, default=1.0)
    parser.add_argument('--eps_end', type=float, default=0.05)
    parser.add_argument('--eps_decay_episodes', type=int, default=300,
                        help='episodes over which epsilon decays linearly from '
                             'eps_start to eps_end.')
    parser.add_argument('--model_path', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0,
                        help='master seed: fixes python/numpy/torch/CUDA + the '
                             'per-episode SUMO vehicle stream for full reproducibility.')
    # ---- reward selection ----
    parser.add_argument('--reward', type=str, default='composite',
                        choices=list(REWARD_MAP.keys()),
                        help="reward function. Built-in single-metric rewards "
                             "(queue/diffwait/pressure) are for the k-hop analysis; "
                             "'composite' is the multi-term reward for the main "
                             "comparison experiment (see composite_reward.py).")
    # ---- evaluation action selection ----
    parser.add_argument('--eval_mode', type=str, default='deterministic',
                        choices=['stochastic', 'deterministic'],
                        help="eval action selection. 'deterministic' (default): "
                             "greedy argmax (the deployed DQN policy). 'stochastic': "
                             "keep eps_end exploration during eval.")
    parser.add_argument('--tripinfo', type=int, default=1,
                        help='1: write & parse SUMO tripinfo at eval to report '
                             'average travel time and throughput.')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    delta_time = 10

    set_seed(args.seed)

    # Results go under ./<reward>_s<seed>/DQN_<net> so IDDQN runs never clobber
    # the hop-indexed IPPO/CMAT runs and sit alongside MAT/MAPPO/HAPPO per
    # (reward, seed). The net suffix keeps mlp / gat side by side.
    run_dir = os.path.abspath(f"./{args.reward}_s{args.seed}/DQN_{args.net}")
    os.makedirs(run_dir, exist_ok=True)
    train_log_path = os.path.join(run_dir, "pretrain_train.txt")
    eval_log_path = os.path.join(run_dir, "pretrain_eval.txt")
    latest_ckpt = os.path.join(run_dir, "pretrain_latest.pth")
    best_ckpt = os.path.join(run_dir, "pretrain_best.pth")

    tripinfo_path = os.path.join(run_dir, "tripinfo.xml") if args.tripinfo else None
    extra_cmd = "--scale 2.0 --no-warnings"
    if tripinfo_path:
        extra_cmd += f" --tripinfo-output {tripinfo_path}"

    reward_fn = make_composite_reward() if args.reward == 'composite' else REWARD_MAP[args.reward]

    env = sumo_rl.parallel_env(
        net_file='../data/raw_data/manhattan_28x7/manhattan_28x7.net.xml',
        route_file='../data/raw_data/manhattan_28x7/manhattan_28x7.rou.xml',
        use_gui=False, num_seconds=args.max_step * delta_time, delta_time=delta_time,
        yellow_time=2, min_green=5, max_green=50,
        reward_fn=reward_fn,
        additional_sumo_cmd=extra_cmd,
    )
    device = torch.device("cuda:" + args.cuda if torch.cuda.is_available() and not args.cpu else 'cpu')

    agent = Agent(env, {}, device, hidden_dim=args.hidden_dim, lr=args.lr,
                  gamma=args.gamma, horizon_len=args.max_step, training_episodes=500,
                  buffer_size=args.buffer_size, net=args.net,
                  use_agent_emb=bool(args.use_agent_emb), n_heads=args.n_heads,
                  target_update_freq=args.target_update_freq, double=bool(args.double))
    agent.set_base_seed(args.seed)

    print(f"[Config] MODEL=IDDQN({args.net.upper()}{' , 1-hop GAT' if args.net=='gat' else ', independent'}) | "
          f"reward={args.reward}({REWARD_MAP[args.reward]}) | seed={args.seed} | "
          f"double={bool(args.double)} | use_agent_emb={bool(args.use_agent_emb)} | "
          f"eps={args.eps_start}->{args.eps_end} over {args.eps_decay_episodes} ep | "
          f"warmup={args.warmup} | eval_mode={args.eval_mode} | tripinfo={bool(args.tripinfo)}")

    if args.model_path:
        agent.load(args.model_path)

    best_reward = -1e8
    loss_list = []
    global_env_steps = 0

    for epoch in range(1, 501):
        # Linearly decayed epsilon for this episode.
        frac = min(1.0, (epoch - 1) / max(1, args.eps_decay_episodes))
        eps = args.eps_start + frac * (args.eps_end - args.eps_start)

        observe = agent.reset(train=True, train_ppo=True, episode=epoch)
        total_reward = 0.0
        ep_losses = []
        pbar = tqdm.tqdm(range(args.max_step))

        for t in pbar:
            # Force pure exploration during warmup so the buffer fills with
            # decorrelated random experience before learning starts.
            step_eps = 1.0 if global_env_steps < args.warmup else eps
            action_dic = agent.act(observe, exploration_rate=step_eps)
            observe, reward, done = agent.step(action_dic)
            total_reward += reward
            global_env_steps += 1

            if global_env_steps >= args.warmup:
                l = agent.learn(batch_size=args.batch_size, updates=args.updates_per_step)
                if l:
                    ep_losses.append(l)
            if done:
                break

        agent.scheduler.step()
        loss = float(np.mean(ep_losses)) if ep_losses else 0.0
        loss_list.append(loss)
        loss = np.mean(loss_list)

        # Travel time / throughput for the TRAINING episode: close so SUMO
        # finalizes the tripinfo file, then parse it.
        if args.tripinfo:
            try:
                agent.env.close()
                tm = parse_tripinfo(tripinfo_path)
                agent.travel_time = tm['travel_time']
                agent.throughput = tm['throughput']
            except Exception as e:
                print(f"[tripinfo] parse skipped: {e}")

        stats = agent.get_stats()
        log = (f"Epoch {epoch} | Reward: {total_reward:.2f} | Loss: {loss:.4f} | Eps: {eps:.3f} | "
               f"TravelTime: {stats['travel_time']:.2f} | Throughput: {stats['throughput']} | "
               f"Queue: {stats['avg_queue']:.2f} | Wait: {stats['avg_wait']:.2f} | "
               f"Speed: {stats['avg_speed']:.2f} | Fairness: {stats['avg_fairness']:.2f}")
        print(log)
        with open(train_log_path, 'a') as f:
            f.write(log + "\n")
        agent.save(latest_ckpt)

        if epoch % args.eval_episode == 0:
            observe = agent.reset(train=False)
            eval_reward = 0.0
            eval_deterministic = (args.eval_mode == 'deterministic')
            eval_eps = 0.0 if eval_deterministic else args.eps_end
            for t in tqdm.tqdm(range(args.max_step)):
                action_dic = agent.act(observe, exploration_rate=eval_eps,
                                       deterministic=eval_deterministic)
                observe, reward, done = agent.step(action_dic)
                eval_reward += reward
                if done:
                    break

            if args.tripinfo:
                try:
                    agent.env.close()
                    tm = parse_tripinfo(tripinfo_path)
                    agent.travel_time = tm['travel_time']
                    agent.throughput = tm['throughput']
                except Exception as e:
                    print(f"[tripinfo] parse skipped: {e}")

            stats = agent.get_stats()
            log = (f"Eval {epoch} | Reward: {eval_reward:.2f} | "
                   f"TravelTime: {stats['travel_time']:.2f} | Throughput: {stats['throughput']} | "
                   f"Queue: {stats['avg_queue']:.2f} | Wait: {stats['avg_wait']:.2f} | "
                   f"Speed: {stats['avg_speed']:.2f} | Fairness: {stats['avg_fairness']:.2f}")
            print(log)
            with open(eval_log_path, 'a') as f:
                f.write(log + "\n")

            if eval_reward > best_reward:
                best_reward = eval_reward
                agent.save(best_ckpt)


if __name__ == '__main__':
    main()
