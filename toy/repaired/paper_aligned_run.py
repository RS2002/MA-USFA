"""Auditable local runner for the paper-aligned synthetic experiments.

The script is intentionally small and serial: one frozen conditional lower-SF
model is trained for a declared kappa, then the same attention upper model is
trained with one of three separately labelled objectives (DDQN, MC-PG, single-trajectory PPO-style PG surrogate).
Only train/validation objectives are loaded.  Test objectives are never read.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import torch

from .exact import evaluate_policy, compose_fixed, legacy_sources
from .legacy_world import SFWorldExact
from .learning import observation
from .paper_aligned_attention import PaperAlignedAttention
from .paper_aligned_lower import (candidate_values, load_context_sf,
                                  train_context_sf)
from .task_protocol import load_tasks, verify_protocol


def _hash_array(a):
    return hashlib.sha256(np.asarray(a).tobytes()).hexdigest()


def _torch_threads():
    n = int(os.environ.get("MA_USFA_THREAD_BUDGET", "2"))
    torch.set_num_threads(min(2, n))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


class EntryReplay:
    def __init__(self, capacity=50000):
        self.capacity = int(capacity); self.size = 0; self.cursor = 0
        self.fields = {
            "state": (np.int64, ()), "time": (np.int16, ()),
            "next_state": (np.int64, ()), "action": (np.int8, ()),
            "done": (np.float32, ()), "reward": (np.float32, ()),
            "weights": (np.float32, (2, 4)),
        }
        self.data = {k: np.zeros((self.capacity,) + sh, dtype=dt)
                     for k, (dt, sh) in self.fields.items()}

    def append(self, **row):
        for k in self.fields: self.data[k][self.cursor] = row[k]
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, rng, batch):
        ix = rng.choice(self.size, int(batch), replace=False)
        return {k: v[ix] for k, v in self.data.items()}


def _source_tensor_inputs(model, states, times, weights, sources, lower):
    states = np.asarray(states, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    weights = np.array(weights, dtype=np.float32, copy=True)
    acts = sources[:, times, states, :].transpose(1, 2, 0).astype(np.int64)
    vals = candidate_values(lower, states, times, weights, sources)
    return (torch.as_tensor(observation(states, times)),
            torch.as_tensor(weights), torch.as_tensor(acts, dtype=torch.long),
            torch.as_tensor(vals, dtype=torch.float32))


def _entry_to_primitive(sources, entry, t, state):
    e0, e1 = int(entry) // 4, int(entry) % 4
    a0 = int(sources[e0, t, state, 0]); a1 = int(sources[e1, t, state, 1])
    return a0 * 5 + a1


def evaluate_attention(model, world, tasks, sources, lower):
    rows = []
    model.eval()
    with torch.no_grad():
        for task in tasks:
            w = np.asarray(task["weights"], dtype=np.float32)
            policy = np.empty((world.horizon, world.S), dtype=np.int64)
            states = np.arange(world.S)
            for t in range(world.horizon):
                tt = np.full(world.S, t, dtype=np.int64)
                inputs = _source_tensor_inputs(model, states, tt,
                                               np.broadcast_to(w, (world.S, 2, 4)),
                                               sources, lower)
                ent = model.greedy_entries(*inputs).cpu().numpy()
                policy[t] = np.array([_entry_to_primitive(sources, e, t, s)
                                      for s, e in enumerate(ent[:, 0] * 4 + ent[:, 1])])
            rows.append({"task_id": task["id"], "kind": task["kind"],
                         "return": evaluate_policy(world, policy, w)["initial_return"]})
    model.train()
    groups = sorted({r["kind"] for r in rows})
    return {"mean_return": float(np.mean([r["return"] for r in rows])),
            "group_means": {g: float(np.mean([r["return"] for r in rows if r["kind"] == g])) for g in groups},
            "rows": rows}


def _ddqn_update(model, target, batch, opt, world, sources, lower, gamma):
    x = _source_tensor_inputs(model, batch["state"], batch["time"], batch["weights"], sources, lower)
    q = model(*x); a = torch.as_tensor(batch["action"], dtype=torch.long)
    pred = q.gather(1, a[:, None]).squeeze(1)
    nt = np.minimum(batch["time"].astype(np.int64) + 1, world.horizon - 1)
    with torch.no_grad():
        nx = _source_tensor_inputs(model, batch["next_state"], nt, batch["weights"], sources, lower)
        na = model.greedy_entries(*nx); ni = na[:, 0] * 4 + na[:, 1]
        nq = target(*nx).gather(1, ni[:, None]).squeeze(1)
        done = torch.as_tensor(batch["done"], dtype=torch.float32)
        y = torch.as_tensor(batch["reward"], dtype=torch.float32) + float(gamma) * (1. - done) * nq
    loss = torch.nn.functional.smooth_l1_loss(pred, y)
    opt.zero_grad(set_to_none=True); loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.); opt.step()
    return float(loss.detach())


def _policy_probs(model, inputs, epsilon):
    logits = model(*inputs).reshape(-1, 16)
    soft = logits.softmax(-1)
    return (1. - epsilon) * soft + epsilon / 16.


def _collect_episode(model, world, task, sources, lower, rng, epsilon, mode):
    w = np.asarray(task["weights"], dtype=np.float32)
    state = world.reset(rng=rng)
    rows = []; total = 0.
    for t in range(world.horizon):
        inputs = _source_tensor_inputs(model, np.array([state]), np.array([t]), w[None], sources, lower)
        if mode == "ddqn":
            with torch.no_grad():
                entry = int(model(*inputs).argmax(-1)[0]) if rng.rand() >= epsilon else int(rng.randint(16))
            old_logp = None
        else:
            with torch.no_grad():
                probs = _policy_probs(model, inputs, epsilon)[0]
                dist = torch.distributions.Categorical(probs=probs)
                entry_t = dist.sample(); entry = int(entry_t)
                old_logp = float(dist.log_prob(entry_t))
        action = _entry_to_primitive(sources, entry, t, state)
        nxt, phi, _ = world.step(state, action, rng=rng)
        reward = float(world.reward(phi, w)); total += reward
        rows.append({"state": state, "time": t, "next_state": nxt, "entry": entry,
                     "reward": reward, "done": float(t + 1 == world.horizon),
                     "old_logp": old_logp, "weights": w})
        state = nxt
    return rows, total


def _update_pg_or_ppo(model, baseline, episodes, opt, bopt, gamma, mode, clip=.2):
    obs = []; weights = []; entries = []; old = []; returns = []; masks = []
    for ep in episodes:
        g = 0.; gs = []
        for row in reversed(ep):
            g = row["reward"] + gamma * g; gs.append(g)
        gs = list(reversed(gs))
        for row, ret in zip(ep, gs):
            obs.append(observation(np.array([row["state"]]), np.array([row["time"]]))[0])
            weights.append(row["weights"]); entries.append(row["entry"])
            old.append(row["old_logp"]); returns.append(ret); masks.append(1.)
    obs_t = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
    w_t = torch.as_tensor(np.asarray(weights), dtype=torch.float32)
    # Lower/SF inputs are not retained in the episode; caller supplies them by
    # recomputing before this function in the public wrapper.
    return obs_t, w_t, np.asarray(entries), np.asarray(old), np.asarray(returns), masks


def train_upper(out_dir, world, sources, lower, train_tasks, val_tasks,
                seed=100, budget=10240, mode="ddqn", batch=64,
                warmup=256, update_every=4, target_every=1000,
                lr=3e-4, eval_every=2560):
    """Train one upper method.  The mode is part of the artifact identity."""
    out = Path(out_dir)
    if any((out / name).exists() for name in ("upper.pt", "summary.json")):
        raise FileExistsError("Refusing to overwrite a previous upper run")
    if mode not in ("ddqn", "pg", "ppo") or int(budget) <= 0:
        raise ValueError("invalid optimizer mode or training budget")
    _torch_threads(); torch.manual_seed(seed)
    rng = np.random.RandomState(seed + 17)
    model = PaperAlignedAttention(gamma=world.gamma, horizon=world.horizon)
    target = PaperAlignedAttention(gamma=world.gamma, horizon=world.horizon)
    target.load_state_dict(model.state_dict())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    baseline = None
    if mode in ("pg", "ppo"):
        # A small baseline is deliberately separate from the attention actor.
        baseline = torch.nn.Sequential(torch.nn.Linear(13, 32), torch.nn.Tanh(), torch.nn.Linear(32, 1))
        bopt = torch.optim.Adam(baseline.parameters(), lr=1e-3)
    replay = EntryReplay()
    steps = updates = 0; best = -float("inf"); best_state = None; logs = []
    while steps < int(budget):
        task = train_tasks[int(rng.randint(len(train_tasks)))]
        if mode == "ddqn":
            w = np.asarray(task["weights"], dtype=np.float32); state = world.reset(rng=rng)
            for t in range(world.horizon):
                eps = max(.02, .2 * (1. - steps / max(1., budget * .8)))
                inp = _source_tensor_inputs(model, np.array([state]), np.array([t]), w[None], sources, lower)
                with torch.no_grad(): q = model(*inp)[0]
                entry = int(rng.randint(16)) if rng.rand() < eps else int(q.argmax())
                action = _entry_to_primitive(sources, entry, t, state)
                nxt, phi, _ = world.step(state, action, rng=rng)
                replay.append(state=state, time=t, next_state=nxt, action=entry,
                              done=float(t + 1 == world.horizon), reward=world.reward(phi, w), weights=w)
                state = nxt; steps += 1
                if replay.size >= max(warmup, batch) and steps % update_every == 0:
                    _ddqn_update(model, target, replay.sample(rng, batch), opt, world, sources, lower, world.gamma)
                    updates += 1
                if updates and updates % target_every == 0: target.load_state_dict(model.state_dict())
                if steps >= budget: break
        else:
            ep, _ = _collect_episode(model, world, task, sources, lower, rng,
                                     epsilon=.05, mode=mode)
            # Compute policy-gradient update from the sampled trajectory.
            states = np.array([r["state"] for r in ep]); times = np.array([r["time"] for r in ep])
            w = np.stack([r["weights"] for r in ep]); entries = np.array([r["entry"] for r in ep])
            inp = _source_tensor_inputs(model, states, times, w, sources, lower)
            logits = model(*inp); dist = torch.distributions.Categorical(probs=(.95 * logits.softmax(-1) + .05 / 16.))
            logp = dist.log_prob(torch.as_tensor(entries, dtype=torch.long))
            gs = []; g = 0.
            for r in reversed(ep): g = r["reward"] + world.gamma * g; gs.append(g)
            ret = torch.as_tensor(list(reversed(gs)), dtype=torch.float32)
            bo = torch.as_tensor(observation(states, times), dtype=torch.float32)
            bv = baseline(torch.cat((bo, torch.as_tensor(w.reshape(len(w), -1), dtype=torch.float32)), -1)).squeeze(-1)
            adv = ret - bv.detach()
            if mode == "pg":
                loss = -(logp * adv).mean()
            else:
                old = torch.as_tensor([r["old_logp"] for r in ep], dtype=torch.float32)
                ratio = torch.exp(logp - old)
                clipped = torch.clamp(ratio, 1. - .2, 1. + .2)
                loss = -torch.min(ratio * adv, clipped * adv).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 10.); opt.step()
            bl = torch.nn.functional.mse_loss(bv, ret); bopt.zero_grad(set_to_none=True); bl.backward(); bopt.step()
            steps += world.horizon; updates += 1
        if steps % eval_every == 0 or steps >= budget:
            val = evaluate_attention(model, world, val_tasks, sources, lower)
            logs.append({"steps": steps, "updates": updates, "validation": val})
            if val["mean_return"] > best:
                best = val["mean_return"]; best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None: model.load_state_dict(best_state)
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "mode": mode, "seed": seed, "budget": budget,
                "best_validation": best, "logs": logs,
                "sources_sha256": _hash_array(sources)}, out / "upper.pt")
    (out / "summary.json").write_text(json.dumps({"mode": mode, "seed": seed, "budget": budget,
        "steps": steps, "updates": updates, "best_validation": best, "logs": logs}, indent=2) + "\n")
    return model, {"mode": mode, "steps": steps, "updates": updates,
                   "best_validation": best, "logs": logs}


def load_or_train_lower(root, world, sources, tasks, seed, steps):
    path = Path(root) / "lower.pt"
    if path.exists():
        model, payload = load_context_sf(path)
        expected = {"seed": int(seed), "steps": int(steps),
                    "kappa": float(world.kappa), "gamma": float(world.gamma),
                    "source_sha256": _hash_array(sources)}
        if any(payload.get(key) != value for key, value in expected.items()):
            raise ValueError("Existing lower checkpoint differs in seed/budget/world/source identity")
        return model
    model, _ = train_context_sf(root, sources, tasks, seed=seed, steps=steps,
                                gamma=world.gamma, kappa=world.kappa)
    return model


def contract(protocol_dir, out_dir, kappa=1.0, seed=100):
    verify_protocol(protocol_dir)
    train = load_tasks(protocol_dir, "train"); val = load_tasks(protocol_dir, "validation")
    world = SFWorldExact(horizon=40, gamma=.95, kappa=kappa)
    base = legacy_sources(SFWorldExact(horizon=40, gamma=.95, kappa=0.0))["policies"]
    sources = np.broadcast_to(base[:, None], (len(base), world.horizon, world.S, 2)).copy()
    model = PaperAlignedAttention()
    weights = np.asarray(val[0]["weights"], dtype=np.float32)
    # Contract probes the upper interface with finite placeholder SF values;
    # lower-SF training is a separate stage.
    obs = torch.as_tensor(observation(np.array([0]), np.array([0])))
    acts = torch.as_tensor(sources[:, 0, 0, :].T[None], dtype=torch.long)
    vals = torch.zeros((1, 2, 4), dtype=torch.float32)
    q = model(obs, torch.as_tensor(weights[None]), acts, vals)
    assert q.shape == (1, 16) and torch.isfinite(q).all()
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    (Path(out_dir) / "contract.json").write_text(json.dumps({"status": "PASS", "mode": "paper_aligned_attention", "kappa": kappa,
        "train_count": len(train), "validation_count": len(val), "sources_sha256": _hash_array(sources)}, indent=2) + "\n")
    return {"status": "PASS", "train_count": len(train), "validation_count": len(val)}


def main():
    """Delegate CLI execution to the guarded, split-aware public entry point.

    The earlier --mode/--lower-dir CLI is intentionally retired. Scientific
    training functions above remain importable for audited matrix execution.
    """
    from .public_run import main as public_main
    return public_main()


if __name__ == "__main__":
    raise SystemExit(main())
