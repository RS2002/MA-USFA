"""Teammate-context-conditioned successor-feature model.

The model is independent of SUMO.  A transition collected under source entry
`z_i` and teammate context `c_-i` updates only that executed profile; terminal
transitions have exactly zero bootstrap.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
try:
    import torch
    from torch import nn
except ImportError:  # local static audit without server dependencies
    torch = None
    nn = object

@dataclass(frozen=True)
class ConditionalSFConfig:
    state_dim: int
    feature_dim: int = 5
    source_count: int = 5
    context_dim: int = 5
    hidden_dim: int = 128
    gamma: float = .99
    def validate(self) -> None:
        for name in ('state_dim','feature_dim','source_count','context_dim','hidden_dim'):
            if int(getattr(self, name)) <= 0:
                raise ValueError(name + ' must be positive')
        if not 0. <= float(self.gamma) <= 1.:
            raise ValueError('gamma must be in [0,1]')

class ConditionalSF(nn.Module if torch is not None else object):
    """Shared per-node predictor chi_i(s | z_i, c_-i)."""
    def __init__(self, config: ConditionalSFConfig):
        if torch is None:
            raise RuntimeError('conditional_sf requires torch on the server')
        config.validate(); super().__init__(); self.config = config
        input_dim = config.state_dim + config.source_count + config.context_dim
        self.net = nn.Sequential(nn.Linear(input_dim, config.hidden_dim), nn.ReLU(),
                                 nn.Linear(config.hidden_dim, config.hidden_dim), nn.ReLU(),
                                 nn.Linear(config.hidden_dim, config.feature_dim))
    def forward(self, state, own_source, teammate_context):
        cfg = self.config
        if state.ndim != 3 or state.shape[-1] != cfg.state_dim:
            raise ValueError(f'state must have shape [B,N,{cfg.state_dim}]')
        if own_source.shape != state.shape[:2]:
            raise ValueError('own_source must have shape [B,N]')
        if teammate_context.shape != (*state.shape[:2], cfg.context_dim):
            raise ValueError('teammate_context must have shape [B,N,context_dim]')
        own_source = own_source.to(dtype=torch.long)
        if bool(((own_source < 0) | (own_source >= cfg.source_count)).any()):
            raise ValueError('own_source contains an invalid source index')
        one_hot = torch.nn.functional.one_hot(own_source, cfg.source_count).to(state.dtype)
        x = torch.cat((state, one_hot, teammate_context.to(state.dtype)), dim=-1)
        result = self.net(x.reshape(-1, x.shape[-1])).reshape(*state.shape[:2], cfg.feature_dim)
        if not bool(torch.isfinite(result).all()):
            raise FloatingPointError('conditional SF produced non-finite values')
        return result

def td_target(features, next_prediction, done, gamma: float):
    """SF target; done=1 yields features exactly, with no bootstrap."""
    if torch is None: raise RuntimeError('td_target requires torch on the server')
    if features.shape != next_prediction.shape: raise ValueError('shape mismatch')
    done = done.to(dtype=features.dtype)
    if done.ndim == 1 and done.shape[0] == features.shape[0]: done = done[:, None, None]
    while done.ndim < features.ndim: done = done.unsqueeze(-1)
    if done.shape[:2] not in (features.shape[:2], (features.shape[0], 1)):
        raise ValueError('done must align [B] or [B,N]')
    target = features + float(gamma) * (1. - done) * next_prediction
    if not bool(torch.isfinite(target).all()): raise FloatingPointError('non-finite TD target')
    return target

def executed_entry_td_loss(prediction, target, executed_source, source_count: int):
    """MSE for executed entries only; targets are never broadcast to K entries."""
    if torch is None: raise RuntimeError('executed_entry_td_loss requires torch on the server')
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError('prediction and target must have shape [B,N,F]')
    if executed_source.shape != prediction.shape[:2]: raise ValueError('executed_source shape mismatch')
    if bool(((executed_source < 0) | (executed_source >= int(source_count))).any()):
        raise ValueError('executed_source contains invalid entry')
    return torch.nn.functional.mse_loss(prediction, target)

def numpy_terminal_target(features, next_prediction, done, gamma: float):
    features = np.asarray(features, dtype=np.float64)
    next_prediction = np.asarray(next_prediction, dtype=np.float64)
    done = np.asarray(done, dtype=np.float64)
    if features.shape != next_prediction.shape: raise ValueError('shape mismatch')
    if done.shape == (features.shape[0],): done = done[:, None, None]
    elif done.shape in (features.shape[:2], features.shape[:2] + (1,)): pass
    else: raise ValueError('done must have shape [B], [B,N] or [B,N,1]')
    while done.ndim < features.ndim: done = done[..., None]
    result = features + float(gamma) * (1. - done) * next_prediction
    if not np.all(np.isfinite(result)): raise FloatingPointError('non-finite TD target')
    return result
