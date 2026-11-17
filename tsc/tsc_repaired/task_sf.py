"""Task-conditioned source-policy SF for the repaired traffic pipeline.

For a query at node i, own source z remains fixed, while at every following
step each other node independently samples a source from D(. | w_j).  The
distribution D is declared in the task protocol.  Actual teammate source
indices are training actions, never SF inputs.  In particular this interface
does not need to know the profile the upper selector is about to choose.

This state SF starts by executing beta_i^z.  It supplies the source-action
slice used by strict selection, rather than an arbitrary first-action SF.
Node observations and attention compress the full traffic state; prediction
accuracy and exact Markov sufficiency are not guaranteed by this interface.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn

TASK_SF_VERSION = "ma-usfa-task-sf-v2"
REFERENCE_VERSION = "ma-usfa-task-reference-v2"
REFERENCE_SEMANTICS = (
    "own source z fixed; teammate sources sampled independently from "
    "D(.|teammate task) at every step; actual teammate entries are not inputs"
)


@dataclass(frozen=True)
class TaskSFConfig:
    state_dim: int
    source_count: int
    feature_dim: int = 5
    task_dim: int = 5
    hidden_dim: int = 64
    attention_dim: int = 32
    gamma: float = 0.99

    def validate(self):
        for name in ("state_dim", "source_count", "feature_dim", "task_dim",
                     "hidden_dim", "attention_dim"):
            if isinstance(getattr(self, name), bool) or int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0.0 <= float(self.gamma) <= 1.0:
            raise ValueError("gamma must lie in [0,1]")


class TaskConditionedSF(nn.Module):
    """Predict chi_i(state | own z, teammate task weights), shape [B,N,F].

    Attention masks self before reading task weights.  The task of node i
    therefore affects its candidate *reward readout* but not its predicted
    successor features.  All complete per-node task vectors remain available
    to the attention operator; they are not replaced by source identities.
    A supplied adjacency limits attention to declared neighboring nodes.
    """

    def __init__(self, config: TaskSFConfig):
        super().__init__()
        config.validate()
        self.config = config
        own_dim = config.state_dim + config.source_count + 1
        self.query = nn.Linear(own_dim, config.attention_dim, bias=False)
        teammate_dim = config.state_dim + config.task_dim
        self.key = nn.Linear(teammate_dim, config.attention_dim, bias=False)
        self.value = nn.Linear(teammate_dim, config.attention_dim, bias=False)
        self.net = nn.Sequential(
            nn.Linear(own_dim + config.attention_dim, config.hidden_dim),
            nn.ReLU(), nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.ReLU(), nn.Linear(config.hidden_dim, config.feature_dim))

    def _time(self, time_fraction, state):
        b, n = state.shape[:2]
        if time_fraction is None:
            raise ValueError("finite-horizon task SF requires time_fraction")
        time = torch.as_tensor(time_fraction, dtype=state.dtype, device=state.device)
        if time.ndim == 0:
            time = time.expand(b)
        if time.shape == (b,):
            time = time[:, None, None].expand(-1, n, -1)
        elif time.shape == (b, n):
            time = time[..., None]
        if time.shape != (b, n, 1):
            raise ValueError("time_fraction must be [B], [B,N] or [B,N,1]")
        if not bool(torch.isfinite(time).all()) or bool(((time < 0) | (time > 1)).any()):
            raise ValueError("time_fraction must be finite and lie in [0,1]")
        return time

    def forward(self, state, own_source, task_weights, time_fraction,
                adjacency=None):
        cfg = self.config
        if state.ndim != 3 or state.shape[-1] != cfg.state_dim:
            raise ValueError(f"state must have shape [B,N,{cfg.state_dim}]")
        if own_source.shape != state.shape[:2]:
            raise ValueError("own_source must have shape [B,N]")
        if task_weights.shape != (*state.shape[:2], cfg.task_dim):
            raise ValueError("task_weights must have shape [B,N,task_dim]")
        if task_weights.device != state.device or own_source.device != state.device:
            raise ValueError("SF inputs must share a device")
        if not bool(torch.isfinite(state).all()) or not bool(torch.isfinite(task_weights).all()):
            raise ValueError("SF state/tasks must be finite")
        if own_source.is_floating_point() and not bool((own_source == own_source.floor()).all()):
            raise ValueError("own_source must contain integer source indices")
        own_source = own_source.to(torch.long)
        if bool(((own_source < 0) | (own_source >= cfg.source_count)).any()):
            raise ValueError("own_source outside the frozen library")
        b, n = state.shape[:2]
        own = torch.cat((state, nn.functional.one_hot(
            own_source, cfg.source_count).to(state.dtype), self._time(time_fraction, state)), -1)
        # Own task weights never enter query.  A diagonal mask removes their
        # key/value rows before any teammate task aggregation.
        teammates = torch.cat((state, task_weights.to(state.dtype)), -1)
        logits = self.query(own) @ self.key(teammates).transpose(-1, -2)
        logits = logits / math.sqrt(cfg.attention_dim)
        allowed = ~torch.eye(n, dtype=torch.bool, device=state.device)[None]
        allowed = allowed.expand(b, -1, -1)
        if adjacency is not None:
            graph = torch.as_tensor(adjacency, dtype=torch.bool, device=state.device)
            if graph.shape == (n, n):
                graph = graph[None].expand(b, -1, -1)
            if graph.shape != (b, n, n):
                raise ValueError("adjacency must have shape [N,N] or [B,N,N]")
            allowed = allowed & graph
        # No-neighbor rows (including N=1) have exactly zero context rather
        # than the NaNs produced by softmax over all -inf entries.
        attention = logits.masked_fill(~allowed, torch.finfo(logits.dtype).min).softmax(-1)
        attention = attention * allowed.to(attention.dtype)
        attention = attention / attention.sum(-1, keepdim=True).clamp_min(
            torch.finfo(attention.dtype).tiny)
        context = attention @ self.value(teammates)
        result = self.net(torch.cat((own, context), -1))
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError("task-conditioned SF produced non-finite features")
        return result

    def candidate_values(self, state, weights, time_fraction, adjacency=None):
        """Return source-action SF reward readouts [B,N,K], without entries.

        NumPy inputs are accepted for the SUMO runtime.  The returned tensor
        stays on the model device and uses the complete five reward features.
        """
        device = next(self.parameters()).device
        state = torch.as_tensor(state, dtype=torch.float32, device=device)
        weights = torch.as_tensor(weights, dtype=torch.float32, device=device)
        if state.ndim != 3:
            raise ValueError("candidate state must have shape [B,N,D]")
        if self.config.feature_dim != self.config.task_dim:
            raise ValueError("linear readout requires matching feature/task dimensions")
        values = []
        for source in range(self.config.source_count):
            own = torch.full(state.shape[:2], source, dtype=torch.long, device=device)
            feature = self(state, own, weights, time_fraction, adjacency)
            values.append((feature * weights).sum(-1))
        return torch.stack(values, -1)


def task_sf_td_target(features, next_prediction, done, gamma):
    """Vector Bellman target with exact zero bootstrap on terminal rows."""
    if features.shape != next_prediction.shape or features.ndim != 3:
        raise ValueError("features and next_prediction must share [B,N,F]")
    d = torch.as_tensor(done, dtype=features.dtype, device=features.device)
    if d.shape == (features.shape[0],):
        d = d[:, None, None]
    elif d.shape == features.shape[:2]:
        d = d[..., None]
    if d.shape not in ((features.shape[0], 1, 1), (*features.shape[:2], 1)):
        raise ValueError("done must have shape [B] or [B,N]")
    if not bool(((d == 0) | (d == 1)).all()):
        raise ValueError("done must contain only zero or one")
    if not 0 <= float(gamma) <= 1:
        raise ValueError("gamma must lie in [0,1]")
    result = features + float(gamma) * torch.where(d.bool(), torch.zeros_like(next_prediction), next_prediction)
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("non-finite task-SF TD target")
    return result


def load_task_sf(path, device="cpu"):
    """Load and freeze a V2 checkpoint; reject legacy entry-context SF."""
    try:
        payload = torch.load(Path(path), map_location=device, weights_only=False)
    except TypeError:
        payload = torch.load(Path(path), map_location=device)
    if not isinstance(payload, dict) or payload.get("version") != TASK_SF_VERSION:
        raise ValueError("SF checkpoint is not task-conditioned V2; legacy SF needs retraining")
    model = TaskConditionedSF(TaskSFConfig(**payload["config"])).to(device)
    model.load_state_dict(payload["state_dict"], strict=True)
    if any(not bool(torch.isfinite(p).all()) for p in model.parameters()):
        raise ValueError("SF checkpoint contains non-finite parameters")
    model.eval().requires_grad_(False)
    return model, payload


def task_sf_checkpoint(model, **metadata):
    return {"version": TASK_SF_VERSION, "config": asdict(model.config),
            "state_dict": model.state_dict(), "reference_semantics": REFERENCE_SEMANTICS,
            "approximation": "local observations and attention-compressed teammate state/tasks",
            **metadata}
