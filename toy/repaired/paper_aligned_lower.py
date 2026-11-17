"""Paper-aligned teammate-task-conditioned lower SF layer.

This module deliberately does not map a teammate task to a named reference
expert.  The lower model receives the teammate task weights directly.  During
SF training, source indices are sampled from a declared library-anchor
distribution D_z(.|w); the sampled source indices are training labels used to
generate trajectories, not observations that must be inferred at deployment.

The model learns action-SF values for a candidate own source z and teammate
task weights w_-i.  The continuation after the current action follows the
sampled source component z, which makes the lower quantity explicit and
testable.  The upper layer uses the source-action slice of this action-SF.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .learning import observation
from .task_protocol import source_weights
from .legacy_world import SFWorldExact


class ContextActionSF(nn.Module):
    """Predict [B,5 actions,5 features] for one agent and one source z."""

    def __init__(self, hidden=64):
        super().__init__()
        # observation(5) + agent id(2) + own source z(4) + teammate task w_-i(4)
        self.input_dim = 15
        self.output_dim = 25
        self.net = nn.Sequential(
            nn.Linear(self.input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, self.output_dim))

    def forward(self, obs, agent, own_source, teammate_weights):
        ai = torch.nn.functional.one_hot(agent.long(), 2).float()
        zi = torch.nn.functional.one_hot(own_source.long(), 4).float()
        x = torch.cat([obs.float(), ai, zi, teammate_weights.float()], dim=-1)
        return self.net(x).reshape(-1, 5, 5)


def source_anchor_probabilities(weights, temperature=0.5):
    """Declared D_z(.|w): preference-weighted softmax over fixed source anchors."""
    w = np.asarray(weights, dtype=np.float64)
    anchors = source_weights().astype(np.float64)  # [4,2,4]
    scores = np.einsum('kif,bif->bik', anchors, w)
    scores = scores / float(temperature)
    scores -= scores.max(-1, keepdims=True)
    p = np.exp(scores)
    return p / p.sum(-1, keepdims=True)


def _sample_task(tasks, rng):
    task = tasks[int(rng.randint(len(tasks)))]
    return np.asarray(task['weights'], dtype=np.float32)


class TransitionReplay:
    fields = {
        'state': (np.int64, ()), 'time': (np.int16, ()),
        'next_state': (np.int64, ()), 'action': (np.int8, ()),
        'done': (np.float32, ()), 'features': (np.float32, (2, 5)),
        'weights': (np.float32, (2, 4)), 'sources': (np.int8, (2,)),
    }

    def __init__(self, capacity=100000):
        self.capacity = int(capacity)
        self.size = 0
        self.cursor = 0
        self.data = {k: np.zeros((self.capacity,) + shape, dtype=dtype)
                     for k, (dtype, shape) in self.fields.items()}

    def append(self, **row):
        for key in self.fields:
            self.data[key][self.cursor] = row[key]
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, rng, batch):
        ix = rng.choice(self.size, int(batch), replace=False)
        return {k: v[ix] for k, v in self.data.items()}


def _query_rows(model, state, times, agent, own_source, teammate_weights):
    obs = torch.as_tensor(observation(state, times), dtype=torch.float32)
    return model(obs, torch.as_tensor(agent, dtype=torch.long),
                 torch.as_tensor(own_source, dtype=torch.long),
                 torch.as_tensor(teammate_weights, dtype=torch.float32))


def candidate_values(model, states, times, weights, sources):
    """Return source-action SF dot products [B,2,4]."""
    states = np.asarray(states, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    weights = np.asarray(weights, dtype=np.float32)
    B = len(states)
    values = np.empty((B, 2, 4), dtype=np.float32)
    with torch.no_grad():
        for agent in range(2):
            rows = B * 4
            own_source = np.tile(np.arange(4), B)
            row_state = np.repeat(states, 4)
            row_time = np.repeat(times, 4)
            teammate = np.repeat(weights[:, 1-agent, :], 4, axis=0)
            agent_ids = np.full(rows, agent, dtype=np.int64)
            pred = _query_rows(model, row_state, row_time, agent_ids,
                               own_source, teammate)
            action = sources[own_source, row_time, row_state, agent]
            feat = pred[np.arange(rows), action]
            own_w = np.repeat(weights[:, agent, :], 4, axis=0)
            theta = np.concatenate([own_w, -2. * np.ones((len(own_w), 1), dtype=np.float32)], axis=-1)
            values[:, agent, :] = (feat * torch.as_tensor(theta)).sum(-1).cpu().numpy().reshape(B, 4)
    return values


def sf_batch_loss(model, target, batch, world, gamma, sources):
    B = len(batch['state'])
    losses = []
    for agent in range(2):
        state = batch['state']
        nxt = batch['next_state']
        time = batch['time']
        z = batch['sources'][:, agent]
        other_w = batch['weights'][:, 1-agent, :]
        own_action = batch['action'] // 5 if agent == 0 else batch['action'] % 5
        obs = torch.as_tensor(observation(state, time), dtype=torch.float32)
        pred_all = model(obs, torch.full((B,), agent), torch.as_tensor(z),
                         torch.as_tensor(other_w))
        pred = pred_all[np.arange(B), own_action]
        next_time = np.minimum(time.astype(np.int64) + 1, world.horizon - 1)
        next_action = sources[z, next_time, nxt, agent]
        next_obs = torch.as_tensor(observation(nxt, next_time), dtype=torch.float32)
        with torch.no_grad():
            boot_all = target(next_obs, torch.full((B,), agent), torch.as_tensor(z),
                              torch.as_tensor(other_w))
            boot = boot_all[np.arange(B), next_action]
            done = torch.as_tensor(batch['done'], dtype=torch.float32).reshape(-1, 1)
            y = torch.as_tensor(batch['features'][:, agent], dtype=torch.float32)
            y = y + float(gamma) * (1. - done) * boot
        losses.append(torch.nn.functional.smooth_l1_loss(pred, y))
    return sum(losses) / len(losses)


def train_context_sf(out_dir, sources, tasks, seed=100, steps=81920,
                     gamma=.95, kappa=1., lr=1e-3, batch_size=128,
                     warmup=512, update_every=4, target_every=1000,
                     epsilon_start=.3, epsilon_final=.05,
                     epsilon_decay_fraction=.8, hidden=64):
    """Train one frozen lower SF model over the declared task distribution."""
    out = Path(out_dir)
    if any((out / name).exists() for name in ("lower.pt", "metadata.json")):
        raise FileExistsError("Refusing to overwrite a previous lower-SF run")
    if int(steps) <= 0:
        raise ValueError("lower steps must be positive")
    out.mkdir(parents=True, exist_ok=True)
    world = SFWorldExact(horizon=40, gamma=gamma, kappa=kappa)
    rng = np.random.RandomState(int(seed))
    torch.manual_seed(int(seed))
    model = ContextActionSF(hidden=hidden)
    target = ContextActionSF(hidden=hidden)
    target.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    replay = TransitionReplay()
    transitions = 0
    updates = 0
    episodes = 0
    while transitions < int(steps):
        W = _sample_task(tasks, rng)
        p = source_anchor_probabilities(W[None], temperature=.5)[0]
        z = np.array([rng.choice(4, p=p[i]) for i in range(2)], dtype=np.int64)
        state = world.reset(rng=rng)
        for t in range(world.horizon):
            greedy = np.array([sources[z[i], t, state, i] for i in range(2)], dtype=np.int64)
            eps = epsilon_start + (epsilon_final - epsilon_start) * min(
                1., transitions / max(1., steps * epsilon_decay_fraction))
            actions = greedy.copy()
            if rng.rand() < eps:
                actions = rng.randint(0, 5, size=2)
            joint = int(actions[0] * 5 + actions[1])
            nxt, feat, _ = world.step(state, actions, rng=rng)
            replay.append(state=state, time=t, next_state=nxt, action=joint,
                          done=float(t + 1 == world.horizon), features=feat,
                          weights=W, sources=z)
            state = nxt
            transitions += 1
            if replay.size >= max(warmup, batch_size) and transitions % update_every == 0:
                batch = replay.sample(rng, batch_size)
                loss = sf_batch_loss(model, target, batch, world, gamma, sources)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                opt.step()
                updates += 1
                if updates % target_every == 0:
                    target.load_state_dict(model.state_dict())
            if transitions >= int(steps):
                break
        episodes += 1
    state = {'model': model.state_dict(), 'target': target.state_dict(),
             'steps': transitions, 'updates': updates, 'episodes': episodes,
             'seed': int(seed), 'gamma': float(gamma), 'kappa': float(kappa),
             'source_sha256': hashlib.sha256(np.asarray(sources).tobytes()).hexdigest(),
             'training_semantics': 'w_minus_i conditioned; source z sampled from D_z(.|w); source continuation after current action'}
    torch.save(state, out / 'lower.pt')
    (out / 'metadata.json').write_text(json.dumps(state, indent=2, default=str) + '\n')
    return model, state


def load_context_sf(path, hidden=64):
    payload = torch.load(path, map_location='cpu', weights_only=False)
    model = ContextActionSF(hidden=hidden)
    model.load_state_dict(payload['model'])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, payload

