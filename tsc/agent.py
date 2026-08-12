import numpy as np
import torch
import torch.nn as nn
from model import MLPQNet, GATQNet, SFQNet, NeighborComposer
from utils import ReplayBuffer
from graph import build_coupling_adjacency, describe_support
from composite_reward import FEATURE_DIM, DEFAULT_W, sample_task, make_source_library


class Agent():
    """IDDQN (Independent Double DQN) training agent.

    Same external interface as the PPO baselines so it plugs into a familiar
    train loop: reset / act / step / get_stats / save / load / set_base_seed,
    with a `.buffer` and `.scheduler`. Learning is OFF-policy, so instead of an
    on-policy ``pretrain`` over a cleared rollout, the caller calls ``learn``
    to draw a minibatch from the persistent replay buffer.

    IDDQN specifics
    ---------------
    * Independent learners: every intersection has its own Q-value (parameters
      shared across agents). The default ``net='mlp'`` scores each agent purely
      from its own observation -- no cross-agent flow. ``net='gat'`` adds a
      single 1-hop graph-attention layer so each agent's Q also depends on its
      immediate neighbours (the real road adjacency).
    * Double DQN target: the greedy next action is chosen by the ONLINE net and
      evaluated by the TARGET net, removing the max-operator overestimation
      bias of vanilla DQN.
    * epsilon-greedy exploration with a linearly decayed epsilon (driven by the
      caller through ``act(..., exploration_rate=eps)``).
    * A hard target-network sync every ``target_update_freq`` gradient steps.

    No k-hop / h-hop, no consensus, no extra tricks -- a faithful independent
    DQN baseline (plus the optional 1-hop GAT coupling requested for the
    comparison).
    """

    def __init__(self, env, component_cache, device, hidden_dim=64, lr=1e-3,
                 gamma=0.99, horizon_len=60, training_episodes=500,
                 buffer_size=100000, net='mlp', use_agent_emb=True, n_heads=4,
                 target_update_freq=200, double=True, grad_clip=1.0, **kwargs):
        assert net in ('mlp', 'gat'), net
        self.env = env
        self.device = device
        self.gamma = gamma
        self.horizon_len = horizon_len
        self.net = net
        self.double = double
        self.grad_clip = grad_clip
        self.target_update_freq = target_update_freq
        self.base_seed = None
        self._learn_steps = 0

        agents_list = env.possible_agents if hasattr(env, 'possible_agents') else env.agents
        self.id = sorted(agents_list)
        self.num = len(self.id)

        self.state_size = env.observation_spaces[self.id[0]].shape[0] + 1  # +timestep
        self.action_size = env.action_spaces[self.id[0]].n

        # ---- build the online + target Q-networks ----
        if net == 'gat':
            # 1-hop coupling adjacency from the real SUMO road topology (self-loops
            # included), shared by the GAT attention mask.
            C, graph_source = build_coupling_adjacency(self.id, env=self.env, method="auto")
            print(f"[Agent] coupling graph source={graph_source}")
            print(f"[Agent] GAT 1-hop adjacency -> {describe_support(C)}")
            net_kwargs = dict(adj=C, hidden_dim=hidden_dim, n_heads=n_heads,
                              use_agent_emb=use_agent_emb)
            self.q_net = GATQNet(self.state_size, self.action_size, self.num, **net_kwargs).to(self.device)
            self.q_target = GATQNet(self.state_size, self.action_size, self.num, **net_kwargs).to(self.device)
        else:
            net_kwargs = dict(hidden_dim=hidden_dim, use_agent_emb=use_agent_emb)
            self.q_net = MLPQNet(self.state_size, self.action_size, self.num, **net_kwargs).to(self.device)
            self.q_target = MLPQNet(self.state_size, self.action_size, self.num, **net_kwargs).to(self.device)

        self.q_target.load_state_dict(self.q_net.state_dict())
        for p in self.q_target.parameters():
            p.requires_grad = False
        self.q_target.eval()

        lr_lambda = lambda step: max(0.0, 1.0 - step / training_episodes)  # linear schedule
        self.optim = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda)

        self.buffer = ReplayBuffer(buffer_size)

        self.epoch_stats = {'queue_sum': 0.0, 'wait_sum': 0.0, 'speed_sum': 0.0,
                            'fairness_sum': 0.0, 'steps': 0}
        self.travel_time = float('nan')
        self.throughput = 0

        print(f"[Agent] MODEL=IDDQN | net={net} | agents={self.num} | "
              f"state={self.state_size} | action={self.action_size} | "
              f"n_embd={hidden_dim} | double={self.double} | "
              f"target_update_freq={self.target_update_freq}")

    # ---- checkpoint I/O ----
    def save(self, path):
        torch.save(self.q_net.state_dict(), path)

    def load(self, path):
        self.q_net.load_state_dict(torch.load(path, map_location=self.device), strict=True)
        self.q_target.load_state_dict(self.q_net.state_dict())

    def set_base_seed(self, seed):
        self.base_seed = seed

    def reset(self, train=True, train_ppo=True, train_td3=False, episode=0):
        self.is_train = train
        self.episode = episode
        self.current_step = 0
        self.epoch_stats = {k: 0.0 for k in self.epoch_stats}
        self.epoch_stats['steps'] = 0
        if self.base_seed is not None:
            env_seed = self.base_seed + episode if train else self.base_seed + 10_000_000
            observe, _ = self.env.reset(seed=env_seed)
        else:
            observe, _ = self.env.reset()
        return observe

    def _build_states(self, observe):
        states_list = [observe[ts] for ts in self.id]
        states = np.array(states_list)
        timestep = np.zeros([states.shape[0], 1]) + self.current_step / self.horizon_len
        return np.concatenate([states, timestep], axis=-1)

    def act(self, observe, exploration_rate=0, deterministic=False):
        """epsilon-greedy action selection over the per-agent Q-values.

        exploration_rate is the epsilon supplied by the training loop (linearly
        decayed). deterministic=True (evaluation) forces greedy argmax.
        """
        states = self._build_states(observe)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,N,s]
        with torch.no_grad():
            q = self.q_net(states_t).squeeze(0)          # [N, A]
            greedy = q.argmax(dim=-1).cpu().numpy()      # [N]

        eps = 0.0 if deterministic else float(exploration_rate)
        if eps > 0.0:
            # Per-agent independent epsilon-greedy: each intersection explores
            # on its own coin flip.
            explore = np.random.rand(self.num) < eps
            rand_actions = np.random.randint(0, self.action_size, size=self.num)
            actions = np.where(explore, rand_actions, greedy)
        else:
            actions = greedy

        action_dic = {ts: int(a) for ts, a in zip(self.id, actions)}
        self.last_states = states
        self.last_actions = actions.astype(np.int64)
        return action_dic

    def step(self, action_dic):
        self.current_step += 1
        next_observe, rewards, dones_dict, _, info = self.env.step(action_dic)
        total_reward = sum(rewards.values())
        reward = [rewards[ts_id] for ts_id in self.id]

        # System-level metrics (broadcast identically into each agent's info).
        sysinfo = {}
        if isinstance(info, dict) and info:
            first = info.get(self.id[0])
            sysinfo = first if isinstance(first, dict) else info
        self.epoch_stats['queue_sum'] += sysinfo.get('system_total_stopped', 0.0)
        self.epoch_stats['wait_sum'] += sysinfo.get('system_mean_waiting_time', 0.0)
        self.epoch_stats['speed_sum'] += sysinfo.get('system_mean_speed', 0.0)
        # Fairness across intersections: dispersion (std) of per-intersection
        # accumulated waiting time over the WHOLE network. The parallel_env
        # wrapper filters each agent's info dict down to keys starting with its
        # own id, so we read each agent's own "<id>_accumulated_waiting_time"
        # from its own info entry and collect across all 196 intersections.
        per_ts_wait = []
        if isinstance(info, dict):
            for a in self.id:
                ainfo = info.get(a)
                if isinstance(ainfo, dict):
                    per_ts_wait.append(ainfo.get(f"{a}_accumulated_waiting_time", 0.0))
        if per_ts_wait:
            self.epoch_stats['fairness_sum'] += float(np.std(per_ts_wait))
        self.epoch_stats['steps'] += 1

        done = (self.current_step >= self.horizon_len) or any(dones_dict.values())

        if self.is_train:
            next_states = self._build_states(next_observe)
            self.buffer.append(
                self.last_states,             # [N, s]
                self.last_actions,            # [N]
                np.asarray(reward, dtype=np.float32),  # [N]
                next_states,                  # [N, s]
                float(done),                  # scalar
            )
        return next_observe, total_reward, done

    def get_stats(self):
        steps = self.epoch_stats['steps']
        if steps == 0:
            base = {'avg_queue': 0.0, 'avg_wait': 0.0, 'avg_speed': 0.0, 'avg_fairness': 0.0}
        else:
            base = {
                'avg_queue': self.epoch_stats['queue_sum'] / steps / self.num,
                'avg_wait': self.epoch_stats['wait_sum'] / steps,
                'avg_speed': self.epoch_stats['speed_sum'] / steps,
                'avg_fairness': self.epoch_stats['fairness_sum'] / steps,
            }
        base['travel_time'] = self.travel_time
        base['throughput'] = self.throughput
        return base

    def learn(self, batch_size=32, updates=1):
        """Run ``updates`` Double-DQN gradient steps on minibatches drawn from
        the replay buffer. Returns the mean TD loss (or 0.0 if the buffer has
        not filled to one batch yet)."""
        if self.buffer.num < batch_size:
            return 0.0

        loss_list = []
        for _ in range(updates):
            states_np, actions_np, rewards_np, next_states_np, dones_np = self.buffer.sample(batch_size)
            states = torch.from_numpy(states_np).float().to(self.device)        # [B,N,s]
            actions = torch.from_numpy(actions_np).long().to(self.device)       # [B,N]
            rewards = torch.from_numpy(rewards_np).float().to(self.device)      # [B,N]
            next_states = torch.from_numpy(next_states_np).float().to(self.device)  # [B,N,s]
            dones = torch.from_numpy(dones_np).float().to(self.device).unsqueeze(-1)  # [B,1]

            # Q(s,a) for the taken actions.
            q = self.q_net(states)                                             # [B,N,A]
            q_taken = q.gather(-1, actions.unsqueeze(-1)).squeeze(-1)          # [B,N]

            with torch.no_grad():
                if self.double:
                    # Double DQN: online net picks the argmax action, target net
                    # scores it -- decoupling selection from evaluation.
                    next_online = self.q_net(next_states)                     # [B,N,A]
                    next_actions = next_online.argmax(dim=-1, keepdim=True)   # [B,N,1]
                    next_q = self.q_target(next_states).gather(-1, next_actions).squeeze(-1)
                else:
                    next_q = self.q_target(next_states).max(dim=-1)[0]        # [B,N]
                # Shared done mask across agents (episode-level termination).
                target = rewards + self.gamma * (1.0 - dones) * next_q        # [B,N]

            loss = torch.mean((q_taken - target) ** 2)
            loss_list.append(loss.item())
            self.optim.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), self.grad_clip)
            self.optim.step()

            # Hard target sync every target_update_freq gradient steps.
            self._learn_steps += 1
            if self._learn_steps % self.target_update_freq == 0:
                self.q_target.load_state_dict(self.q_net.state_dict())

        return float(np.mean(loss_list)) if loss_list else 0.0


class SFReplayBuffer(ReplayBuffer):
    """Replay buffer that additionally stores the per-agent feature vector phi
    observed at the NEXT state (needed for the SF-TD target phi + gamma*psi).

    Each transition: state [N,s], action [N], phi2 [N,K], next_state [N,s],
    done (scalar). The scalar reward is NOT stored -- under SF the reward is
    reconstructed as phi2 . w for whatever task w the update uses (w-free USFA)."""

    def append(self, state, action, phi2, next_state, done):
        super().append(state, action, phi2, next_state, done)

    def sample(self, batch_size):
        states, actions, phi2, next_states, dones = super().sample(batch_size)
        return states, actions, phi2, next_states, dones


class SFAgent:
    """Successor-Feature agent for the TSC transfer experiment.

    Two roles, selected by ``mode``:

    * ``mode='usfa'`` -- train w-free successor features psi with an SF-TD
      update. A fresh task vector w is drawn every episode (set on the reward
      fn via ``reward_fn.set_w``), so psi learns the feature-based value without
      committing to any task. The closed-form transfer value for ANY task is
      then ``Q = psi . w`` (no retraining) -- the free-region mechanism. This
      same mode, evaluated on a fixed target task, is the L2 (per-agent GPI)
      operator: a_i = argmax_a psi_i(s_i,a) . w_i, fully decentralized.
    * ``mode='l1'`` -- SYNCHRONIZED ANCHOR operator. Uses the SAME frozen psi
      plus a small discrete source library {w^1..w^K}. Each step the whole team
      commits to ONE shared source index k*(s) = argmax_k sum_i psi_i(s_i,a^k_i)
      . w_test_i, where a^k_i = argmax_a psi_i(s_i,a) . w^k is source k's own
      greedy action (closed form from psi). O(K*N) per step, NO A^N enumeration,
      so it is feasible at 196 agents. Anchor selection needs the global state
      to compute the team sum, so L1 shares ONE integer (the anchor index) --
      mildly centralized, but never a joint action.
    * ``mode='composer'`` -- freeze (or warm-start from) a trained psi and
      fine-tune a :class:`NeighborComposer` on a SPECIFIC task (w, coupled
      regime) using per-agent TD on that task's reward. This is the paid-region
      recovery method; it is NEIGHBOR-LIMITED (1-hop road graph), never global.

    Same reset/act/step/learn/save/load interface as :class:`Agent`, so it plugs
    into the existing train loop. Action selection is per-agent argmax over the
    transfer value (usfa/L2) or the composer utilities (composer): decentralized
    execution in both cases.
    """

    def __init__(self, env, reward_fn, device, mode='usfa', hidden_dim=64,
                 lr=1e-3, gamma=0.99, horizon_len=60, training_episodes=500,
                 buffer_size=100000, n_heads=4, use_agent_emb=True,
                 target_update_freq=200, grad_clip=1.0, task_scale=(0.5, 1.5),
                 sf_ckpt=None, source_library=None, **kwargs):
        assert mode in ('usfa', 'l1', 'composer'), mode
        self.env = env
        self.reward_fn = reward_fn
        self.device = device
        self.mode = mode
        self.gamma = gamma
        self.horizon_len = horizon_len
        self.grad_clip = grad_clip
        self.target_update_freq = target_update_freq
        self.task_scale = task_scale
        self.base_seed = None
        self._learn_steps = 0
        self.K = FEATURE_DIM

        agents_list = env.possible_agents if hasattr(env, 'possible_agents') else env.agents
        self.id = sorted(agents_list)
        self.num = len(self.id)
        self.state_size = env.observation_spaces[self.id[0]].shape[0] + 1  # +timestep
        self.action_size = env.action_spaces[self.id[0]].n

        # 1-hop road adjacency (shared by the composer's neighbour mixing).
        C, graph_source = build_coupling_adjacency(self.id, env=self.env, method="auto")
        print(f"[SFAgent] coupling graph source={graph_source} -> {describe_support(C)}")
        self.adj = C

        # ---- successor-feature backbone (+ target copy for SF-TD) ----
        self.sf_net = SFQNet(self.state_size, self.action_size, self.K, self.num,
                             hidden_dim=hidden_dim, use_agent_emb=use_agent_emb).to(device)
        self.sf_target = SFQNet(self.state_size, self.action_size, self.K, self.num,
                                hidden_dim=hidden_dim, use_agent_emb=use_agent_emb).to(device)
        if sf_ckpt is not None:
            self.sf_net.load_state_dict(torch.load(sf_ckpt, map_location=device))
        self.sf_target.load_state_dict(self.sf_net.state_dict())
        for p in self.sf_target.parameters():
            p.requires_grad = False
        self.sf_target.eval()

        # current task vector w (per-agent identical unless heterogeneous tasks)
        self.w = DEFAULT_W.copy()

        # L1 source library (only used in mode='l1'): [K, FEATURE_DIM]. Sourced
        # from the frozen psi in closed form -- no extra training.
        if mode == 'l1':
            lib = make_source_library() if source_library is None else np.asarray(
                source_library, dtype=np.float32)
            self.source_library = lib                  # [K, K_feat]
            self.n_sources = lib.shape[0]

        if mode == 'composer':
            # Freeze the SF backbone: the composer only learns the
            # neighbour-limited residual on top of the fixed transfer value.
            for p in self.sf_net.parameters():
                p.requires_grad = False
            self.sf_net.eval()
            self.composer = NeighborComposer(
                self.sf_net, self.adj, self.K, hidden_dim=hidden_dim,
                n_heads=n_heads, use_agent_emb=use_agent_emb).to(device)
            self.composer_target = NeighborComposer(
                self.sf_net, self.adj, self.K, hidden_dim=hidden_dim,
                n_heads=n_heads, use_agent_emb=use_agent_emb).to(device)
            self.composer_target.load_state_dict(self.composer.state_dict())
            for p in self.composer_target.parameters():
                p.requires_grad = False
            self.composer_target.eval()
            # optimize ONLY the residual params (encoder / gat / head), never
            # the shared frozen sf_net backbone.
            self._resid_params = [p for n, p in self.composer.named_parameters()
                                  if not n.startswith('sf_net.') and p.requires_grad]
            params = self._resid_params
        else:
            params = self.sf_net.parameters()

        self.optim = torch.optim.Adam(params, lr=lr)
        lr_lambda = lambda step: max(0.0, 1.0 - step / training_episodes)
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optim, lr_lambda=lr_lambda)
        self.buffer = SFReplayBuffer(buffer_size)

        self.epoch_stats = {'queue_sum': 0.0, 'wait_sum': 0.0, 'speed_sum': 0.0,
                            'fairness_sum': 0.0, 'steps': 0}
        self.travel_time = float('nan')
        self.throughput = 0
        print(f"[SFAgent] MODE={mode} | agents={self.num} | state={self.state_size} "
              f"| action={self.action_size} | feat={self.K}")

    # ---- checkpoint I/O ----
    def save(self, path):
        if self.mode == 'composer':
            torch.save({'sf': self.sf_net.state_dict(),
                        'composer': self.composer.state_dict()}, path)
        else:
            torch.save(self.sf_net.state_dict(), path)

    def save_sf(self, path):
        torch.save(self.sf_net.state_dict(), path)

    def load(self, path):
        blob = torch.load(path, map_location=self.device)
        if isinstance(blob, dict) and 'sf' in blob:
            self.sf_net.load_state_dict(blob['sf'])
            if self.mode == 'composer':
                self.composer.load_state_dict(blob['composer'])
        else:
            self.sf_net.load_state_dict(blob)
        self.sf_target.load_state_dict(self.sf_net.state_dict())

    def set_base_seed(self, seed):
        self.base_seed = seed

    def _w_tensor(self):
        """Per-agent task-weight tensor [1, N, K] for the current task."""
        w = np.tile(self.w, (self.num, 1)) if self.w.ndim == 1 else self.w
        return torch.tensor(w, dtype=torch.float32, device=self.device).unsqueeze(0)

    def reset(self, train=True, episode=0, **kwargs):
        self.is_train = train
        self.episode = episode
        self.current_step = 0
        self.epoch_stats = {k: 0.0 for k in self.epoch_stats}
        self.epoch_stats['steps'] = 0
        # w-free USFA: draw a fresh task each TRAIN episode and push it to the
        # reward fn so the environment reward matches the psi being learned.
        if self.mode == 'usfa' and train:
            rng = np.random.RandomState((self.base_seed or 0) + episode)
            self.w = sample_task(rng, scale=self.task_scale)
            if hasattr(self.reward_fn, 'set_w'):
                self.reward_fn.set_w(self.w)
        if self.base_seed is not None:
            env_seed = self.base_seed + episode if train else self.base_seed + 10_000_000
            observe, _ = self.env.reset(seed=env_seed)
        else:
            observe, _ = self.env.reset()
        return observe

    def _build_states(self, observe):
        states = np.array([observe[ts] for ts in self.id])
        timestep = np.zeros([states.shape[0], 1]) + self.current_step / self.horizon_len
        return np.concatenate([states, timestep], axis=-1)

    def _utilities(self, states_t):
        """Per-agent action values used for control: closed-form transfer
        (usfa/L2) or composed utilities (composer). states_t: [1,N,s].
        (L1 does not use this path -- see _l1_actions.)"""
        w_t = self._w_tensor()
        if self.mode == 'composer':
            return self.composer(states_t, w_t)            # [1,N,A]
        return self.sf_net.q_values(states_t, w_t)         # [1,N,A]

    def _l1_actions(self, states_t):
        """Synchronized-anchor action selection. states_t: [1,N,s].

        For each source k: a^k_i = argmax_a psi_i(s_i,a).w^k  (source's own
        greedy action, closed form). The team commits to the single shared
        anchor k* = argmax_k sum_i psi_i(s_i, a^k_i).w_test_i, then every agent
        plays its a^{k*}_i. Returns [N] int actions. O(K*N), no A^N."""
        psi = self.sf_net(states_t).squeeze(0)             # [N, A, K_feat]
        w_test = torch.tensor(
            np.tile(self.w, (self.num, 1)) if self.w.ndim == 1 else self.w,
            dtype=torch.float32, device=self.device)       # [N, K_feat]
        lib = torch.tensor(self.source_library, dtype=torch.float32,
                           device=self.device)             # [K, K_feat]
        # source-k per-agent greedy actions: argmax_a psi_i(s_i,a).w^k
        # q_src[k,i,a] = sum_f psi[i,a,f] * lib[k,f]
        q_src = torch.einsum('naf,kf->kna', psi, lib)      # [K, N, A]
        a_src = q_src.argmax(dim=-1)                        # [K, N] greedy action
        # team TEST value of playing source-k's actions: sum_i psi_i(s_i,a^k_i).w_test_i
        # gather psi at a_src -> [K, N, K_feat]
        idx = a_src.unsqueeze(-1).expand(-1, -1, self.K)   # [K, N, K_feat]
        psi_kn = psi.unsqueeze(0).expand(self.n_sources, -1, -1, -1)  # [K,N,A,K_feat]
        psi_sel = psi_kn.gather(2, idx.unsqueeze(2)).squeeze(2)       # [K, N, K_feat]
        team_val = torch.einsum('knf,nf->k', psi_sel, w_test)        # [K]
        k_star = int(team_val.argmax().item())
        return a_src[k_star].cpu().numpy().astype(np.int64)          # [N]

    def act(self, observe, exploration_rate=0, deterministic=False):
        states = self._build_states(observe)
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            if self.mode == 'l1':
                greedy = self._l1_actions(states_t)        # [N]
            else:
                u = self._utilities(states_t).squeeze(0)   # [N,A]
                greedy = u.argmax(dim=-1).cpu().numpy()
        eps = 0.0 if deterministic else float(exploration_rate)
        if eps > 0.0:
            explore = np.random.rand(self.num) < eps
            rand_actions = np.random.randint(0, self.action_size, size=self.num)
            actions = np.where(explore, rand_actions, greedy)
        else:
            actions = greedy
        self.last_states = states
        self.last_actions = actions.astype(np.int64)
        return {ts: int(a) for ts, a in zip(self.id, actions)}

    def _collect_phi2(self):
        """Read the per-agent NEXT-state feature vector phi from the reward fn's
        feature cache (populated during env.step by make_sf_reward)."""
        cache = getattr(self.reward_fn, 'feature_cache', {})
        return np.array([cache.get(ts, np.zeros(self.K, np.float32))
                         for ts in self.id], dtype=np.float32)  # [N,K]

    def step(self, action_dic):
        self.current_step += 1
        next_observe, rewards, dones_dict, _, info = self.env.step(action_dic)
        total_reward = sum(rewards.values())

        sysinfo = {}
        if isinstance(info, dict) and info:
            first = info.get(self.id[0])
            sysinfo = first if isinstance(first, dict) else info
        self.epoch_stats['queue_sum'] += sysinfo.get('system_total_stopped', 0.0)
        self.epoch_stats['wait_sum'] += sysinfo.get('system_mean_waiting_time', 0.0)
        self.epoch_stats['speed_sum'] += sysinfo.get('system_mean_speed', 0.0)
        self.epoch_stats['steps'] += 1

        done = (self.current_step >= self.horizon_len) or any(dones_dict.values())
        if self.is_train:
            next_states = self._build_states(next_observe)
            phi2 = self._collect_phi2()
            self.buffer.append(self.last_states, self.last_actions, phi2,
                               next_states, float(done))
        return next_observe, total_reward, done

    def get_stats(self):
        steps = self.epoch_stats['steps']
        if steps == 0:
            base = {'avg_queue': 0.0, 'avg_wait': 0.0, 'avg_speed': 0.0, 'avg_fairness': 0.0}
        else:
            base = {'avg_queue': self.epoch_stats['queue_sum'] / steps / self.num,
                    'avg_wait': self.epoch_stats['wait_sum'] / steps,
                    'avg_speed': self.epoch_stats['speed_sum'] / steps,
                    'avg_fairness': self.epoch_stats['fairness_sum'] / steps}
        base['travel_time'] = self.travel_time
        base['throughput'] = self.throughput
        return base

    def learn(self, batch_size=32, updates=1):
        if self.buffer.num < batch_size:
            return 0.0
        return (self._learn_composer(batch_size, updates) if self.mode == 'composer'
                else self._learn_usfa(batch_size, updates))

    def _learn_usfa(self, batch_size, updates):
        """SF-TD on the w-free features: for the CURRENT episode's task w,
            psi(s,a).w  <-  phi2.w + gamma (1-done) psi_target(s2, a2*).w
        with a2* = argmax_a psi_target(s2,a).w. The vector SF regression target
        is phi2 + gamma psi_target(s2,a2*); we regress psi(s,a) to it per-feature
        (standard USFA), so the same psi serves every task."""
        losses = []
        w_t = self._w_tensor().squeeze(0)                  # [N,K]
        for _ in range(updates):
            S, A, PHI2, S2, D = self.buffer.sample(batch_size)
            S = torch.from_numpy(S).float().to(self.device)     # [B,N,s]
            A = torch.from_numpy(A).long().to(self.device)      # [B,N]
            PHI2 = torch.from_numpy(PHI2).float().to(self.device)  # [B,N,K]
            S2 = torch.from_numpy(S2).float().to(self.device)
            D = torch.from_numpy(D).float().to(self.device).view(-1, 1, 1)  # [B,1,1]

            psi = self.sf_net(S)                                 # [B,N,A,K]
            idx = A.view(*A.shape, 1, 1).expand(-1, -1, 1, self.K)
            psi_taken = psi.gather(2, idx).squeeze(2)            # [B,N,K]

            with torch.no_grad():
                psi2 = self.sf_target(S2)                        # [B,N,A,K]
                q2 = torch.einsum('bnak,nk->bna', psi2, w_t)     # [B,N,A]
                a2 = q2.argmax(dim=-1)                           # [B,N]
                idx2 = a2.view(*a2.shape, 1, 1).expand(-1, -1, 1, self.K)
                psi2_sel = psi2.gather(2, idx2).squeeze(2)       # [B,N,K]
                target = PHI2 + self.gamma * (1.0 - D) * psi2_sel  # [B,N,K]

            loss = torch.mean((psi_taken - target) ** 2)
            losses.append(loss.item())
            self.optim.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self.sf_net.parameters(), self.grad_clip)
            self.optim.step()
            self._learn_steps += 1
            if self._learn_steps % self.target_update_freq == 0:
                self.sf_target.load_state_dict(self.sf_net.state_dict())
        return float(np.mean(losses)) if losses else 0.0

    def _learn_composer(self, batch_size, updates):
        """Per-agent honest TD on the composer utilities for the fixed task w:
            delta_i = r_i + gamma (1-done) max_a u_target(s2,a) - u(s,a_i)
        with r_i = phi2_i . w_i (reconstructed from the stored feature vector).
        Per-agent (not joint-IGM) targets: the joint max overestimates under
        transition coupling. The composer backbone psi is frozen; only the
        neighbour-limited residual learns. Target uses a hard-synced copy."""
        losses = []
        w_t = self._w_tensor()                             # [1,N,K]
        w_bn = w_t.squeeze(0)                               # [N,K]
        for _ in range(updates):
            S, A, PHI2, S2, D = self.buffer.sample(batch_size)
            S = torch.from_numpy(S).float().to(self.device)
            A = torch.from_numpy(A).long().to(self.device)
            PHI2 = torch.from_numpy(PHI2).float().to(self.device)  # [B,N,K]
            S2 = torch.from_numpy(S2).float().to(self.device)
            D = torch.from_numpy(D).float().to(self.device).view(-1, 1)  # [B,1]

            B = S.shape[0]
            w_batch = w_t.expand(B, -1, -1)                 # [B,N,K]
            u = self.composer(S, w_batch)                  # [B,N,A]
            u_taken = u.gather(-1, A.unsqueeze(-1)).squeeze(-1)  # [B,N]
            r = torch.einsum('bnk,nk->bn', PHI2, w_bn)      # [B,N] per-agent reward

            with torch.no_grad():
                u2 = self.composer_target(S2, w_batch)      # [B,N,A]
                u2_max = u2.max(dim=-1)[0]                  # [B,N]
                target = r + self.gamma * (1.0 - D) * u2_max

            loss = torch.mean((u_taken - target) ** 2)
            losses.append(loss.item())
            self.optim.zero_grad()
            loss.backward()
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(self._resid_params, self.grad_clip)
            self.optim.step()
            self._learn_steps += 1
            if self._learn_steps % self.target_update_freq == 0:
                self.composer_target.load_state_dict(self.composer.state_dict())
        return float(np.mean(losses)) if losses else 0.0
