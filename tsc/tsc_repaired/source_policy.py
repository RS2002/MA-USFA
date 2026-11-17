"""Small trainable source-policy network used by the server runner.

Each source is trained from scratch on one frozen anchor task.  The resulting
network is frozen before it is exposed to the strict-entry selector.  The
selector chooses a source index; this module only turns that source index and
the current observation into the source primitive action.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn


class SourceQNet(nn.Module):
    """Shared per-node Q network; no residual action head."""

    def __init__(self, observation_dim: int, action_count: int, hidden: int = 128):
        super().__init__()
        if min(observation_dim, action_count, hidden) <= 0:
            raise ValueError("source network dimensions must be positive")
        self.observation_dim = int(observation_dim)
        self.action_count = int(action_count)
        self.net = nn.Sequential(
            nn.Linear(self.observation_dim + 1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.action_count),
        )

    def forward(self, observations: torch.Tensor, time_fraction: torch.Tensor) -> torch.Tensor:
        if observations.ndim != 3 or observations.shape[-1] != self.observation_dim:
            raise ValueError("observations must have shape [B,N,observation_dim]")
        if time_fraction.ndim == 1:
            time_fraction = time_fraction[:, None, None].expand(-1, observations.shape[1], -1)
        elif time_fraction.ndim == 2:
            time_fraction = time_fraction[:, :, None]
        if time_fraction.shape != (*observations.shape[:2], 1):
            raise ValueError("time_fraction must have shape [B], [B,N] or [B,N,1]")
        x = torch.cat((observations, time_fraction.to(observations.dtype)), dim=-1)
        return self.net(x.reshape(-1, x.shape[-1])).reshape(
            observations.shape[0], observations.shape[1], self.action_count
        )


@dataclass(frozen=True)
class SourceCheckpoint:
    path: Path
    source_index: int
    observation_dim: int
    action_count: int
    hidden: int

    def load(self, device: str = "cpu") -> SourceQNet:
        payload = torch.load(self.path, map_location=device, weights_only=False)
        required = {"state_dict", "observation_dim", "action_count", "hidden"}
        missing = required - set(payload)
        if missing:
            raise ValueError(f"source checkpoint is missing: {sorted(missing)}")
        if int(payload["source_index"]) != self.source_index:
            raise ValueError("source checkpoint index differs from manifest")
        model = SourceQNet(
            int(payload["observation_dim"]), int(payload["action_count"]), int(payload["hidden"])
        )
        model.load_state_dict(payload["state_dict"], strict=True)
        model = model.to(device)
        model.eval()
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return model


def observation_array(observation: dict, agent_ids: Sequence[str]) -> np.ndarray:
    rows = []
    for agent_id in agent_ids:
        if agent_id not in observation:
            raise KeyError(f"observation missing controlled node {agent_id!r}")
        value = np.asarray(observation[agent_id], dtype=np.float32).reshape(-1)
        if value.size == 0 or not np.isfinite(value).all():
            raise ValueError(f"invalid observation for node {agent_id!r}")
        rows.append(value)
    result = np.stack(rows).astype(np.float32, copy=False)
    return result


def greedy_source_action(model: SourceQNet, observations: np.ndarray,
                         time_index: int, horizon: int, device: str = "cpu") -> np.ndarray:
    values = torch.as_tensor(observations[None], dtype=torch.float32, device=device)
    fraction = torch.full((1,), float(time_index) / max(1, int(horizon)), dtype=torch.float32, device=device)
    with torch.no_grad():
        action = model(values, fraction).argmax(-1)[0].cpu().numpy()
    return action.astype(np.int64, copy=False)
