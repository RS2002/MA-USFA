"""Paper-aligned attention upper composer.

The upper controller receives the complete task weights, candidate primitive
actions from the frozen source library, and the conditional lower-SF values.
There is deliberately no dominant-goal/reference-source lookup.  The model
returns one score for each legal pair of source entries (4 x 4).
"""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class PaperAlignedAttention(nn.Module):
    def __init__(self, hidden=32, attn_dim=16, rho=4.0,
                 gamma=.95, horizon=40):
        super().__init__()
        if hidden <= 0 or attn_dim <= 0 or rho <= 0:
            raise ValueError("hidden, attn_dim and rho must be positive")
        self.hidden = int(hidden)
        self.attn_dim = int(attn_dim)
        self.rho = float(rho)
        self.gamma = float(gamma)
        self.horizon = int(horizon)
        self.value_input_scale = (float(horizon) if gamma == 1 else
                                  (1. - gamma ** horizon) / (1. - gamma))
        # 2 coordinates + remaining time + role(2) + own task(4) + teammate
        # task(4) + candidate action one-hot(20) + candidate SF(4).
        self.token_dim = 37
        self.query = nn.Linear(self.token_dim, attn_dim, bias=False)
        self.key = nn.Linear(self.token_dim, attn_dim, bias=False)
        self.value = nn.Linear(self.token_dim, attn_dim, bias=False)
        self.residual_head = nn.Sequential(nn.Linear(attn_dim, hidden), nn.ReLU(),
                                           nn.Linear(hidden, 4))
        self.mixer = nn.Sequential(nn.Linear(2 * self.token_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, 2))
        # Start from the frozen source/SF entry-greedy controller.
        nn.init.zeros_(self.residual_head[-1].weight)
        nn.init.zeros_(self.residual_head[-1].bias)
        nn.init.zeros_(self.mixer[-1].weight)
        nn.init.zeros_(self.mixer[-1].bias)

    def spec(self):
        return {
            "version": "paper-aligned-attention-v1",
            "backbone": "single-round scaled dot-product attention",
            "reference_lookup": False,
            "inputs": "complete task weights, candidate source actions, conditional lower SF",
            "strict_entry_execution": True,
            "parameters": sum(p.numel() for p in self.parameters()),
            "training_algorithms": ["team Double DQN", "MC policy gradient", "single-trajectory PPO-style PG surrogate"],
        }

    @staticmethod
    def _check(obs, weights, actions, values):
        if obs.ndim != 2 or obs.shape[1] != 5:
            raise ValueError("obs must have shape [B,5]")
        expected = (len(obs), 2, 4)
        if tuple(weights.shape) != expected or tuple(actions.shape) != expected or tuple(values.shape) != expected:
            raise ValueError("weights/actions/values must have shape [B,2,4]")
        if not (obs.device == weights.device == actions.device == values.device):
            raise ValueError("all inputs must share a device")
        if not all(torch.isfinite(x).all() for x in (obs, weights, values)):
            raise ValueError("inputs must be finite")

    def tokens(self, obs, weights, actions, values):
        self._check(obs, weights, actions, values)
        b, dtype, device = len(obs), obs.dtype, obs.device
        coords = obs[:, :4].reshape(b, 2, 2)
        rem = obs[:, 4:5].unsqueeze(1).expand(-1, 2, -1)
        roles = torch.eye(2, dtype=dtype, device=device).unsqueeze(0).expand(b, -1, -1)
        own = weights
        other = weights.flip(1)
        act = F.one_hot(actions.long(), 5).to(dtype).reshape(b, 2, 20)
        vals = values / self.value_input_scale
        return torch.cat((coords, rem, roles, own, other, act, vals), dim=-1)

    def components(self, obs, weights, actions, values):
        tokens = self.tokens(obs, weights, actions, values)
        q = self.query(tokens)
        k = self.key(tokens)
        v = self.value(tokens)
        attn = (q @ k.transpose(-1, -2) / math.sqrt(self.attn_dim)).softmax(-1)
        raw = self.residual_head(attn @ v)
        base = (values[:, 0, :, None] + values[:, 1, None, :]).reshape(-1, 16)
        utilities = values + self.rho * torch.tanh(raw)
        mixing = 2. * self.mixer(tokens.reshape(b := len(tokens), -1)).softmax(-1)
        mixing = mixing.clamp_min(torch.finfo(mixing.dtype).tiny)
        mixing = 2. * mixing / mixing.sum(-1, keepdim=True)
        joint = (mixing[:, 0, None, None] * utilities[:, 0, :, None]
                 + mixing[:, 1, None, None] * utilities[:, 1, None, :]).reshape(-1, 16)
        if not torch.isfinite(joint).all():
            raise FloatingPointError("nonfinite attention scores")
        return {"tokens": tokens, "attention": attn, "raw": raw,
                "base": base, "utilities": utilities, "mixing": mixing,
                "joint_q": joint}

    def forward(self, obs, weights, actions, values):
        return self.components(obs, weights, actions, values)["joint_q"]

    def greedy_entries(self, obs, weights, actions, values):
        q = self.forward(obs, weights, actions, values)
        best = q.argmax(-1)
        return torch.stack((best // 4, best % 4), -1)

