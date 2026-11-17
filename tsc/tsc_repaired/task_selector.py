"""Topology-masked attention selector and executed-entry Double DQN for traffic.

Scores are additive over nodes once state, tasks and candidates are fixed.
Consequently per-node greedy source selection exactly maximizes this model's
joint score, without enumerating K**N profiles. This is a model-class fact;
it is not a guarantee about the true traffic return or DDQN convergence.
"""
from __future__ import annotations

from collections import deque
from dataclasses import asdict, dataclass
import copy
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

SELECTOR_VERSION = "ma-usfa-traffic-attention-ddqn-v2"


@dataclass(frozen=True)
class TaskSelectorConfig:
    observation_dim: int
    source_count: int
    action_count: int
    task_dim: int = 5
    hidden_dim: int = 64
    attention_heads: int = 4
    gamma: float = 0.99
    horizon: int = 60
    rho: float = 4.0

    def validate(self) -> None:
        for key in ("observation_dim", "source_count", "action_count", "task_dim",
                    "hidden_dim", "attention_heads", "horizon"):
            if int(getattr(self, key)) <= 0:
                raise ValueError(f"{key} must be positive")
        if self.hidden_dim % self.attention_heads:
            raise ValueError("hidden_dim must be divisible by attention_heads")
        if not 0. <= self.gamma <= 1. or not np.isfinite(self.rho) or self.rho <= 0.:
            raise ValueError("gamma must be in [0,1] and rho must be finite and positive")


class TaskAttentionSelector(nn.Module):
    """One topology-masked attention round; team-value training is centralized.

    Candidate actions and task weights are available before selection.
    Queries retain the local SF readout. Neighbor keys/values contain only
    raw local inputs, so a neighbor never rebroadcasts its already aggregated
    SF. Stacking the lower and upper layers therefore does not add a second
    task-information hop.
    No actually selected teammate source index is an input. The team score
    and training reward both use the mean over nodes. Reported rollout team
    returns must separately sum actual node rewards and time steps.
    """

    def __init__(self, config: TaskSelectorConfig):
        super().__init__()
        config.validate()
        self.config = config
        token_dim = (config.observation_dim + config.task_dim + 1
                     + config.source_count * config.action_count + config.source_count)
        self.token_dim = token_dim
        self.encoder = nn.Sequential(nn.Linear(token_dim, config.hidden_dim),
                                     nn.LayerNorm(config.hidden_dim), nn.ReLU())
        self.attention = nn.MultiheadAttention(config.hidden_dim, config.attention_heads,
                                              dropout=0., batch_first=True)
        self.residual_head = nn.Sequential(nn.Linear(config.hidden_dim, config.hidden_dim),
                                          nn.ReLU(), nn.Linear(config.hidden_dim, config.source_count))
        self.mixer_head = nn.Linear(config.hidden_dim, 1)
        self.register_buffer("_adjacency", torch.empty((0, 0), dtype=torch.bool), persistent=False)
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.mixer_head.weight)
        nn.init.zeros_(self.mixer_head.bias)
        self.value_input_scale = (float(config.horizon) if config.gamma == 1. else
                                  (1. - config.gamma ** config.horizon) / (1. - config.gamma))

    def set_adjacency(self, adjacency):
        mask = torch.as_tensor(adjacency, dtype=torch.bool, device=next(self.parameters()).device)
        if mask.ndim != 2 or mask.shape[0] != mask.shape[1] or mask.shape[0] == 0:
            raise ValueError("adjacency must be a non-empty square [N,N] mask")
        if not bool(mask.diagonal().all()):
            raise ValueError("adjacency must include every node itself")
        self._adjacency = mask.clone()
        return self

    def _check(self, obs, weights, candidate_actions, sf_values, time_fraction):
        cfg = self.config
        if obs.ndim != 3 or obs.shape[-1] != cfg.observation_dim:
            raise ValueError("obs must be [B,N,observation_dim]")
        b, n, _ = obs.shape
        if b <= 0 or n <= 0 or weights.shape != (b, n, cfg.task_dim):
            raise ValueError("task_weights must share [B,N] and task_dim")
        if candidate_actions.shape != (b, n, cfg.source_count) or sf_values.shape != candidate_actions.shape:
            raise ValueError("candidate actions/SF must be [B,N,K]")
        if not (obs.device == weights.device == candidate_actions.device == sf_values.device):
            raise ValueError("all selector inputs must share a device")
        for name, tensor in (("obs", obs), ("weights", weights), ("sf_values", sf_values)):
            if not bool(torch.isfinite(tensor).all()):
                raise ValueError(f"{name} contains non-finite values")
        if candidate_actions.dtype not in (torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8):
            raise ValueError("candidate actions must have integer dtype")
        if bool(((candidate_actions < 0) | (candidate_actions >= cfg.action_count)).any()):
            raise ValueError("candidate actions contain an invalid primitive action")
        if time_fraction is None:
            times = obs.new_zeros((b, n, 1))
        else:
            times = time_fraction
            if times.ndim == 1:
                times = times[:, None, None].expand(b, n, 1)
            elif times.ndim == 2:
                times = times[:, :, None]
            if times.shape != (b, n, 1) or times.device != obs.device or not bool(torch.isfinite(times).all()):
                raise ValueError("time_fraction must be finite [B], [B,N], or [B,N,1]")
        return times

    def tokens(self, obs, weights, candidate_actions, sf_values, time_fraction=None):
        times = self._check(obs, weights, candidate_actions, sf_values, time_fraction)
        actions = F.one_hot(candidate_actions.long(), self.config.action_count).to(obs.dtype)
        actions = actions.flatten(start_dim=-2)
        return torch.cat((obs, weights.to(obs.dtype), times.to(obs.dtype), actions,
                          sf_values.to(obs.dtype) / self.value_input_scale), dim=-1)

    def components(self, obs, weights, candidate_actions, sf_values, time_fraction=None):
        tokens = self.tokens(obs, weights, candidate_actions, sf_values, time_fraction)
        encoded = self.encoder(tokens)
        # Lower SF at node j already includes its neighbors' tasks. Sending
        # SF_j through the upper attention to node i would expand the total
        # input radius to two hops. Keys and values therefore contain only
        # raw node-local state, own task, time and candidate source actions.
        # Queries and the local utility may still use SF_i.
        neighbor_tokens = torch.cat((tokens[..., :-self.config.source_count],
                                     torch.zeros_like(tokens[..., -self.config.source_count:])), -1)
        neighbor_encoded = self.encoder(neighbor_tokens)
        mask = None
        if self._adjacency.numel():
            if self._adjacency.shape != (obs.shape[1], obs.shape[1]):
                raise ValueError("adjacency node count differs from the current observation")
            mask = ~self._adjacency
        attended, _ = self.attention(encoded, neighbor_encoded, neighbor_encoded,
                                     attn_mask=mask, need_weights=False)
        # The learned correction starts at zero: the initialized greedy policy
        # is exactly source-entry greedy under the fixed first-index tie rule.
        residual = self.config.rho * torch.tanh(self.residual_head(encoded + attended))
        utilities = sf_values + residual
        # Positive, source-choice-independent local mixing; zero logits give 1.
        logits = self.mixer_head(encoded + attended).squeeze(-1)
        mixing = (F.softplus(logits) / float(np.log(2.))).clamp_min(torch.finfo(logits.dtype).tiny)
        if not bool(torch.isfinite(utilities).all()) or not bool(torch.isfinite(mixing).all()):
            raise FloatingPointError("selector produced non-finite scores")
        return {"utilities": utilities, "mixing": mixing, "tokens": tokens,
                "neighbor_tokens": neighbor_tokens}

    def forward(self, obs, weights, candidate_actions, sf_values, time_fraction=None):
        """Per-node utilities [B,N,K]; mixing is in :meth:`components`."""
        return self.components(obs, weights, candidate_actions, sf_values, time_fraction)["utilities"]

    def choose_entries(self, obs, weights, candidate_actions, sf_values, time_fraction=None):
        return self(obs, weights, candidate_actions, sf_values, time_fraction).argmax(-1)

    def q_for_entries(self, obs, weights, candidate_actions, sf_values, time_fraction, entries):
        parts = self.components(obs, weights, candidate_actions, sf_values, time_fraction)
        if entries.shape != obs.shape[:2] or entries.dtype != torch.long:
            raise ValueError("entries must be long [B,N]")
        if bool(((entries < 0) | (entries >= self.config.source_count)).any()):
            raise ValueError("entries contain an invalid source index")
        chosen = parts["utilities"].gather(-1, entries[..., None]).squeeze(-1)
        return (parts["mixing"] * chosen).mean(-1)

    def spec(self):
        return {"version": SELECTOR_VERSION, "config": asdict(self.config),
                "backbone": ("one-round topology-masked attention" if self._adjacency.numel() else "global attention diagnostic; no topology supplied"),
                "strict_entry_execution": True, "actual_teammate_entries_input": False,
                "neighbor_sf_rebroadcast": False,
                "neighbor_key_value_inputs": "raw local observation, own task, time, source actions; no SF",
                "local_query_inputs": "raw local inputs and own task-conditioned SF readout",
                "team_score_reduction": "mean of positive-weighted selected node utilities",
                "training_reward_reduction": "mean node reward",
                "parameters": sum(parameter.numel() for parameter in self.parameters())}


@dataclass(frozen=True)
class TaskTransition:
    obs: np.ndarray
    weights: np.ndarray
    candidate_actions: np.ndarray
    sf_values: np.ndarray
    time_fraction: float
    entries: np.ndarray
    reward: float
    next_obs: np.ndarray
    next_candidate_actions: np.ndarray
    next_sf_values: np.ndarray
    next_time_fraction: float
    done: bool


def bounded_replay_capacity(requested: int, prototype: TaskTransition, max_bytes: int,
                            batch_size: int) -> tuple[int, int]:
    """Conservatively cap replay using array bytes plus Python/batch overhead."""
    if requested <= 0 or batch_size <= 0 or max_bytes <= 0:
        raise ValueError("replay capacity, batch size and max bytes must be positive")
    arrays = (prototype.obs, prototype.weights, prototype.candidate_actions,
              prototype.sf_values, prototype.entries, prototype.next_obs,
              prototype.next_candidate_actions, prototype.next_sf_values)
    # Array objects and duplicated tensor batches are not free. Reserve four
    # batch copies plus 25% slack rather than counting only payload bytes.
    bytes_per_row = int(sum(np.asarray(value).nbytes for value in arrays) * 1.25) + 2048
    usable_rows = max_bytes // bytes_per_row - 4 * batch_size
    capacity = min(int(requested), int(usable_rows))
    if capacity < batch_size:
        raise ValueError("resource limit is too small for one replay batch")
    return capacity, bytes_per_row


class TaskDDQNTrainer:
    """Online replay updates; targets label only the executed joint profile."""

    def __init__(self, model: TaskAttentionSelector, *, seed=0, lr=3e-4,
                 target_every=100, replay_capacity=1024, device="cpu"):
        if target_every <= 0 or replay_capacity <= 0:
            raise ValueError("target_every and replay_capacity must be positive")
        self.device = torch.device(device)
        self.online = model.to(self.device)
        self.target = copy.deepcopy(model).to(self.device).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=float(lr))
        self.replay = deque(maxlen=int(replay_capacity))
        self.rng = np.random.default_rng(int(seed))
        self.target_every = int(target_every)
        self.updates = 0

    def add(self, row: TaskTransition):
        self.replay.append(row)

    def _inputs(self, rows, next_state=False):
        def stack(name, dtype=torch.float32):
            return torch.as_tensor(np.stack([getattr(row, name) for row in rows]), dtype=dtype, device=self.device)
        return (stack("next_obs" if next_state else "obs"), stack("weights"),
                stack("next_candidate_actions" if next_state else "candidate_actions", torch.long),
                stack("next_sf_values" if next_state else "sf_values"),
                torch.as_tensor([row.next_time_fraction if next_state else row.time_fraction for row in rows],
                                dtype=torch.float32, device=self.device))

    def targets(self, rows):
        rewards = torch.as_tensor([row.reward for row in rows], dtype=torch.float32, device=self.device)
        done = torch.as_tensor([row.done for row in rows], dtype=torch.bool, device=self.device)
        if not bool(torch.isfinite(rewards).all()):
            raise ValueError("replay rewards are non-finite")
        # Terminal rows never query the next network. Besides exact zero
        # bootstrap, this prevents irrelevant terminal NaNs entering targets.
        result = rewards.clone()
        active = (~done).nonzero(as_tuple=False).flatten()
        if len(active):
            active_rows = [rows[int(index)] for index in active.cpu().numpy()]
            with torch.no_grad():
                next_inputs = self._inputs(active_rows, next_state=True)
                entries = self.online.choose_entries(*next_inputs)
                next_q = self.target.q_for_entries(*next_inputs, entries)
                result[active] += self.online.config.gamma * next_q
        return result

    def update(self, batch_size=16):
        if len(self.replay) < int(batch_size) or int(batch_size) <= 0:
            raise ValueError("replay is smaller than the requested positive batch size")
        indices = self.rng.choice(len(self.replay), size=int(batch_size), replace=False)
        rows = [self.replay[int(index)] for index in indices]
        inputs = self._inputs(rows)
        entries = torch.as_tensor(np.stack([row.entries for row in rows]), dtype=torch.long, device=self.device)
        q = self.online.q_for_entries(*inputs, entries)
        target = self.targets(rows)
        if not bool(torch.isfinite(target).all()):
            raise FloatingPointError("non-finite DDQN target")
        loss = F.smooth_l1_loss(q, target)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.online.parameters(), 10.)
        if not bool(torch.isfinite(grad_norm)):
            raise FloatingPointError("non-finite DDQN gradient")
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.target_every == 0:
            self.target.load_state_dict(self.online.state_dict())
        return {"loss": float(loss.detach().cpu()), "mean_q": float(q.detach().mean().cpu()),
                "mean_target": float(target.detach().mean().cpu()), "grad_norm": float(grad_norm),
                "updates": self.updates}


def load_selector(path: str | Path, device="cpu") -> tuple[TaskAttentionSelector, dict[str, Any]]:
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, dict) or payload.get("version") != SELECTOR_VERSION:
        raise ValueError("not a task-conditioned traffic selector V2 checkpoint; legacy selector rejected")
    config = TaskSelectorConfig(**payload["config"])
    model = TaskAttentionSelector(config).to(device)
    model.load_state_dict(payload["online"], strict=True)
    if "adjacency" not in payload:
        raise ValueError("selector checkpoint is missing its topology mask")
    model.set_adjacency(payload["adjacency"])
    spec = payload.get("spec")
    if not isinstance(spec, dict) or spec.get("neighbor_sf_rebroadcast") is not False:
        raise ValueError("selector checkpoint lacks the one-hop no-SF-rebroadcast contract; retraining is required")
    model.eval().requires_grad_(False)
    return model, payload
