"""Isolated component experiments for the repaired synthetic MA-USFA model.

The existing paper-aligned implementation is kept unchanged.  Training uses
the same source library, task splits, transition budgets and DDQN recipe.
This module never loads a task split; its caller supplies train/validation
tasks and the final frozen evaluator separately supplies held-out tasks.

``no_attention`` removes cross-token attention, not all teammate information:
tokens still contain teammate weights and the team mixer sees both tokens.
``no_teammate_conditioning`` removes teammate weights only from lower SF;
the upper controller still receives the complete task weights.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import numpy as np
import torch
from torch import nn

from .exact import evaluate_policy
from .legacy_world import SFWorldExact
from .learning import observation
from .paper_aligned_attention import PaperAlignedAttention
from .paper_aligned_lower import (
    ContextActionSF, TransitionReplay, _sample_task, candidate_values,
    load_context_sf, sf_batch_loss, source_anchor_probabilities,
)
from .paper_aligned_run import EntryReplay, _entry_to_primitive, _torch_threads


VERSION = "icml-component-ablation-v1"
VARIANTS = (
    "entry_greedy", "no_attention", "no_sf", "no_teammate_conditioning",
    "attention_control",
)


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_sha256(array):
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def model_sha256(model):
    if model is None:
        return None
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode())
        a = tensor.detach().cpu().contiguous().numpy()
        h.update(str(a.dtype).encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def _json_write(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _torch_write(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def _task_digest(tasks):
    raw = json.dumps(tasks, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode()).hexdigest()


def _protocol_identity(protocol_dir, train_tasks, val_tasks):
    out = {"train_tasks_sha256": _task_digest(train_tasks),
           "validation_tasks_sha256": _task_digest(val_tasks)}
    if protocol_dir is not None:
        directory = Path(protocol_dir)
        # Deliberately do not read or fingerprint the sealed split here.
        out["protocol_manifest_sha256"] = file_sha256(directory / "manifest.json")
        out["train_file_sha256"] = file_sha256(directory / "train.json")
        out["validation_file_sha256"] = file_sha256(directory / "validation.json")
    return out


class AblationAttention(PaperAlignedAttention):
    def spec(self):
        spec = super().spec()
        spec.update({"version": VERSION,
                     "variant": getattr(self, "experiment_variant", "attention_control"),
                     "training_algorithms": ["team Double DQN"]})
        return spec


class EntryGreedy(nn.Module):
    """No learned upper layer; choose the highest-scoring frozen SF entries."""

    def __init__(self, gamma=.95, horizon=40):
        super().__init__()
        self.gamma, self.horizon = float(gamma), int(horizon)

    _check = staticmethod(PaperAlignedAttention._check)
    greedy_entries = PaperAlignedAttention.greedy_entries

    def forward(self, obs, weights, actions, values):
        self._check(obs, weights, actions, values)
        return (values[:, 0, :, None] + values[:, 1, None, :]).reshape(-1, 16)

    def spec(self):
        return {"version": VERSION, "variant": "entry_greedy",
                "backbone": "frozen lower source-entry greedy; no upper optimization",
                "strict_entry_execution": True, "parameters": 0,
                "tie_breaking": "first joint source-entry index",
                "comparison": "same lower and candidate actions as paired attention DDQN"}


class SelfOnlyAttention(AblationAttention):
    """Original token/value/head/mixer with the attention matrix fixed to I."""

    def components(self, obs, weights, actions, values):
        tokens = self.tokens(obs, weights, actions, values)
        attention = torch.eye(2, dtype=tokens.dtype, device=tokens.device)
        attention = attention.unsqueeze(0).expand(len(tokens), -1, -1)
        raw = self.residual_head(self.value(tokens))
        utilities = values + self.rho * torch.tanh(raw)
        mixing = 2. * self.mixer(tokens.reshape(len(tokens), -1)).softmax(-1)
        mixing = mixing.clamp_min(torch.finfo(mixing.dtype).tiny)
        mixing = 2. * mixing / mixing.sum(-1, keepdim=True)
        joint = (mixing[:, 0, None, None] * utilities[:, 0, :, None]
                 + mixing[:, 1, None, None] * utilities[:, 1, None, :]).reshape(-1, 16)
        if not torch.isfinite(joint).all():
            raise FloatingPointError("Nonfinite self-only attention scores")
        return {"tokens": tokens, "attention": attention, "raw": raw,
                "base": (values[:, 0, :, None] + values[:, 1, None, :]).reshape(-1, 16),
                "utilities": utilities, "mixing": mixing, "joint_q": joint}

    def spec(self):
        spec = super().spec()
        spec.update({"version": VERSION, "variant": "no_attention",
                     "backbone": "self-only value projection; attention fixed to identity",
                     "unused_parameters": sum(p.numel() for m in (self.query, self.key)
                                              for p in m.parameters()),
                     "teammate_information": "teammate task in tokens; both tokens in mixer"})
        spec["active_parameters"] = spec["parameters"] - spec["unused_parameters"]
        return spec


class NoSFAttention(AblationAttention):
    """Learn source-entry utilities without SF inputs or an SF value base.

    Utilities are unbounded.  Retaining rho*tanh after removing the SF base
    would artificially restrict the team Q range to +/-2rho.
    """

    def __init__(self, hidden=32, attn_dim=16, rho=4., gamma=.95, horizon=40):
        super().__init__(hidden=hidden, attn_dim=attn_dim, rho=rho,
                         gamma=gamma, horizon=horizon)
        self.token_dim = 33
        self.query = nn.Linear(self.token_dim, attn_dim, bias=False)
        self.key = nn.Linear(self.token_dim, attn_dim, bias=False)
        self.value = nn.Linear(self.token_dim, attn_dim, bias=False)
        self.mixer = nn.Sequential(nn.Linear(2 * self.token_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, 2))
        nn.init.zeros_(self.mixer[-1].weight)
        nn.init.zeros_(self.mixer[-1].bias)

    def tokens(self, obs, weights, actions, values):
        # Values are accepted only to share the public model interface; neither
        # their contents nor a lower-SF checkpoint are used by this variant.
        self._check(obs, weights, actions, values)
        b, dtype, device = len(obs), obs.dtype, obs.device
        coords = obs[:, :4].reshape(b, 2, 2)
        rem = obs[:, 4:5].unsqueeze(1).expand(-1, 2, -1)
        roles = torch.eye(2, dtype=dtype, device=device).unsqueeze(0).expand(b, -1, -1)
        act = torch.nn.functional.one_hot(actions.long(), 5).to(dtype).reshape(b, 2, 20)
        return torch.cat((coords, rem, roles, weights, weights.flip(1), act), -1)

    def components(self, obs, weights, actions, values):
        tokens = self.tokens(obs, weights, actions, values)
        q, k, v = self.query(tokens), self.key(tokens), self.value(tokens)
        attention = (q @ k.transpose(-1, -2) / self.attn_dim ** .5).softmax(-1)
        raw = self.residual_head(attention @ v)
        mixing = 2. * self.mixer(tokens.reshape(len(tokens), -1)).softmax(-1)
        mixing = mixing.clamp_min(torch.finfo(mixing.dtype).tiny)
        mixing = 2. * mixing / mixing.sum(-1, keepdim=True)
        joint = (mixing[:, 0, None, None] * raw[:, 0, :, None]
                 + mixing[:, 1, None, None] * raw[:, 1, None, :]).reshape(-1, 16)
        if not torch.isfinite(joint).all():
            raise FloatingPointError("Nonfinite no-SF attention scores")
        return {"tokens": tokens, "attention": attention, "raw": raw,
                "base": torch.zeros_like(joint), "utilities": raw,
                "mixing": mixing, "joint_q": joint}

    def spec(self):
        spec = super().spec()
        spec.update({"version": VERSION, "variant": "no_sf",
                     "inputs": "complete task weights and candidate source actions; no SF",
                     "token_dim": self.token_dim,
                     "utilities": "unbounded learned raw utility; zero initialized",
                     "lower_training_cost": 0,
                     "parameters": sum(p.numel() for p in self.parameters())})
        return spec


class NoTeammateContextSF(ContextActionSF):
    """Same action-SF TD objective, with no teammate task in its inputs."""

    def __init__(self, hidden=64):
        super().__init__(hidden=hidden)
        self.input_dim = 11
        self.net = nn.Sequential(nn.Linear(self.input_dim, hidden), nn.ReLU(),
                                 nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, self.output_dim))

    def forward(self, obs, agent, own_source, teammate_weights):
        ai = torch.nn.functional.one_hot(agent.long(), 2).float()
        zi = torch.nn.functional.one_hot(own_source.long(), 4).float()
        return self.net(torch.cat((obs.float(), ai, zi), -1)).reshape(-1, 5, 5)


def make_ablation_model(variant, gamma=.95, horizon=40):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown ablation variant: {variant}")
    cls = {"entry_greedy": EntryGreedy, "no_attention": SelfOnlyAttention,
           "no_sf": NoSFAttention}.get(variant, AblationAttention)
    model = cls(gamma=gamma, horizon=horizon)
    model.experiment_variant = variant
    return model


def ablation_tensor_inputs(model, states, times, weights, sources, lower):
    states = np.asarray(states, dtype=np.int64)
    times = np.asarray(times, dtype=np.int64)
    weights = np.array(weights, dtype=np.float32, copy=True)
    actions = sources[:, times, states, :].transpose(1, 2, 0).astype(np.int64)
    if isinstance(model, NoSFAttention):
        values = np.zeros((len(states), 2, 4), dtype=np.float32)
    else:
        if lower is None:
            raise ValueError("This variant requires the paired frozen lower SF")
        values = candidate_values(lower, states, times, weights, sources)
    return (torch.as_tensor(observation(states, times)), torch.as_tensor(weights),
            torch.as_tensor(actions, dtype=torch.long),
            torch.as_tensor(values, dtype=torch.float32))


def evaluate_ablation(model, world, tasks, sources, lower):
    """Exact finite-horizon policy evaluation, with no optimizer updates."""
    was_training = model.training
    model.eval()
    rows = []
    try:
        with torch.no_grad():
            for task in tasks:
                weights = np.asarray(task["weights"], dtype=np.float32)
                policy = np.empty((world.horizon, world.S), dtype=np.int64)
                states = np.arange(world.S)
                for t in range(world.horizon):
                    inputs = ablation_tensor_inputs(model, states, np.full(world.S, t),
                        np.broadcast_to(weights, (world.S, 2, 4)), sources, lower)
                    entries = model.greedy_entries(*inputs).cpu().numpy()
                    policy[t] = [ _entry_to_primitive(sources, e, t, s)
                                  for s, e in enumerate(entries[:, 0] * 4 + entries[:, 1])]
                value = evaluate_policy(world, policy, weights)["initial_return"]
                rows.append({"task_id": task["id"], "kind": task["kind"],
                             "return": float(value)})
    finally:
        model.train(was_training)
    if not rows:
        raise ValueError("Evaluation needs at least one task")
    return {"mean_return": float(np.mean([r["return"] for r in rows])),
            "group_means": {g: float(np.mean([r["return"] for r in rows if r["kind"] == g]))
                            for g in sorted({r["kind"] for r in rows})},
            "rows": rows}


def _ddqn_update(model, target, batch, opt, world, sources, lower):
    x = ablation_tensor_inputs(model, batch["state"], batch["time"],
                              batch["weights"], sources, lower)
    action = torch.as_tensor(batch["action"], dtype=torch.long)
    prediction = model(*x).gather(1, action[:, None]).squeeze(1)
    next_times = np.minimum(batch["time"].astype(np.int64) + 1, world.horizon - 1)
    with torch.no_grad():
        nx = ablation_tensor_inputs(model, batch["next_state"], next_times,
                                   batch["weights"], sources, lower)
        entries = model.greedy_entries(*nx)
        next_action = entries[:, 0] * 4 + entries[:, 1]
        next_q = target(*nx).gather(1, next_action[:, None]).squeeze(1)
        done = torch.as_tensor(batch["done"], dtype=torch.float32)
        y = torch.as_tensor(batch["reward"], dtype=torch.float32)
        y = y + world.gamma * (1. - done) * next_q
    loss = torch.nn.functional.smooth_l1_loss(prediction, y)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
    opt.step()
    return float(loss.detach())


def train_ablation_upper(out_dir, world, sources, lower, train_tasks, val_tasks,
                         variant, seed=100, budget=81920, batch=64,
                         warmup=256, update_every=4, target_every=1000,
                         lr=3e-4, eval_every=2560, lower_checkpoint=None,
                         protocol_dir=None, on_progress: Callable | None = None):
    """Same transition/update/validation recipe as paper_aligned_run.train_upper.

    Existing output files are never resumed or overwritten.  Intermediate
    best checkpoints are marked RUNNING so a partial run cannot be mistaken
    for a complete budget.  The caller provides resource supervision.
    """
    if variant == "entry_greedy":
        raise ValueError("entry_greedy has no upper training; evaluate its frozen model")
    if int(budget) <= 0 or int(eval_every) <= 0:
        raise ValueError("budget and eval_every must be positive")
    out = Path(out_dir)
    if any((out / name).exists() for name in ("upper.pt", "summary.json", "progress.json")):
        raise FileExistsError(f"Refusing to overwrite or silently resume {out}")
    out.mkdir(parents=True, exist_ok=True)
    _torch_threads()
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed + 17)
    model = make_ablation_model(variant, gamma=world.gamma, horizon=world.horizon)
    target = make_ablation_model(variant, gamma=world.gamma, horizon=world.horizon)
    target.load_state_dict(model.state_dict())
    initial_model_hash = model_sha256(model)
    if lower is not None:
        lower.eval()
        for parameter in lower.parameters():
            parameter.requires_grad_(False)
    if variant == "no_teammate_conditioning" and not isinstance(lower, NoTeammateContextSF):
        raise TypeError("no_teammate_conditioning needs its separately trained 11-input lower")
    frozen_lower_hash = model_sha256(lower)
    source_hash = array_sha256(sources)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    replay = EntryReplay()
    steps = updates = 0
    best, best_step = -float("inf"), None
    best_state = None
    logs = []
    identity = _protocol_identity(protocol_dir, train_tasks, val_tasks)
    identity.update({"sources_sha256": source_hash,
                     "lower_state_sha256": frozen_lower_hash,
                     "lower_sha256": file_sha256(lower_checkpoint) if lower_checkpoint else None,
                     "code_sha256": file_sha256(__file__)})
    recipe = {"batch": batch, "warmup": warmup, "update_every": update_every,
              "target_every": target_every, "lr": lr, "eval_every": eval_every,
              "epsilon": "max(.02,.2*(1-steps/(budget*.8)))", "replay_capacity": 50000,
              "upper_transition_budget": int(budget),
              "lower_transition_budget_charged": 0 if variant == "no_sf" else None,
              "checkpoint_selection": "one checkpoint by mean over all validation tasks"}

    def write(status):
        summary = {"status": status, "version": VERSION, "variant": variant,
                   "mode": "ddqn", "seed": int(seed), "budget": int(budget),
                   "steps": steps, "updates": updates, "best_validation": best,
                   "best_step": best_step, "logs": logs, "recipe": recipe,
                   "model_spec": model.spec(), "gamma": world.gamma,
                   "horizon": world.horizon, "kappa": world.kappa,
                   "initial_model_sha256": initial_model_hash, "test_accessed": False,
                   **identity}
        payload = dict(summary)
        payload["model"] = best_state
        _torch_write(out / "upper.pt", payload)
        _json_write(out / ("summary.json" if status == "COMPLETE" else "progress.json"), summary)
        if status == "COMPLETE":
            _json_write(out / "progress.json", summary)
        return summary

    while steps < int(budget):
        task = train_tasks[int(rng.randint(len(train_tasks)))]
        weights = np.asarray(task["weights"], dtype=np.float32)
        state = world.reset(rng=rng)
        for t in range(world.horizon):
            epsilon = max(.02, .2 * (1. - steps / max(1., budget * .8)))
            inputs = ablation_tensor_inputs(model, np.array([state]), np.array([t]),
                                            weights[None], sources, lower)
            with torch.no_grad():
                q = model(*inputs)[0]
                entry = int(rng.randint(16)) if rng.rand() < epsilon else int(q.argmax())
            primitive = _entry_to_primitive(sources, entry, t, state)
            nxt, features, _ = world.step(state, primitive, rng=rng)
            replay.append(state=state, time=t, next_state=nxt, action=entry,
                          done=float(t + 1 == world.horizon),
                          reward=world.reward(features, weights), weights=weights)
            state = nxt
            steps += 1
            if replay.size >= max(warmup, batch) and steps % update_every == 0:
                loss = _ddqn_update(model, target, replay.sample(rng, batch), optimizer,
                                    world, sources, lower)
                if not np.isfinite(loss):
                    raise FloatingPointError("Nonfinite DDQN loss")
                updates += 1
            if updates and updates % target_every == 0:
                target.load_state_dict(model.state_dict())
            if steps >= budget:
                break
        if steps % eval_every == 0 or steps >= budget:
            validation = evaluate_ablation(model, world, val_tasks, sources, lower)
            logs.append({"steps": steps, "updates": updates, "validation": validation})
            if validation["mean_return"] > best:
                best, best_step = validation["mean_return"], steps
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            progress = write("RUNNING")
            if on_progress is not None:
                on_progress(progress)
    if array_sha256(sources) != source_hash or model_sha256(lower) != frozen_lower_hash:
        raise RuntimeError("Frozen source or lower changed during upper training")
    model.load_state_dict(best_state)
    summary = write("COMPLETE")
    return model, summary


def train_ablation_lower(out_dir, sources, tasks, seed=100, steps=81920,
                         gamma=.95, kappa=1., lr=1e-3, batch_size=128,
                         warmup=512, update_every=4, target_every=1000,
                         epsilon_start=.3, epsilon_final=.05,
                         epsilon_decay_fraction=.8, hidden=64, horizon=40,
                         protocol_dir=None, on_progress: Callable | None = None):
    """Retrain lower SF without teammate weights, using original reference TD."""
    out = Path(out_dir)
    if any((out / name).exists() for name in ("lower.pt", "metadata.json", "progress.json")):
        raise FileExistsError(f"Refusing to overwrite or silently resume {out}")
    out.mkdir(parents=True, exist_ok=True)
    _torch_threads()
    world = SFWorldExact(horizon=horizon, gamma=gamma, kappa=kappa)
    if sources.shape != (4, world.horizon, world.S, 2):
        raise ValueError("Source table shape does not match world")
    source_hash = array_sha256(sources)
    rng = np.random.RandomState(int(seed))
    torch.manual_seed(int(seed))
    model = NoTeammateContextSF(hidden=hidden)
    target = NoTeammateContextSF(hidden=hidden)
    target.load_state_dict(model.state_dict())
    initial_hash = model_sha256(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    replay = TransitionReplay()
    transitions = updates = episodes = 0
    while transitions < int(steps):
        weights = _sample_task(tasks, rng)
        probabilities = source_anchor_probabilities(weights[None], temperature=.5)[0]
        z = np.array([rng.choice(4, p=probabilities[i]) for i in range(2)], dtype=np.int64)
        state = world.reset(rng=rng)
        for t in range(world.horizon):
            actions = np.array([sources[z[i], t, state, i] for i in range(2)], dtype=np.int64)
            epsilon = epsilon_start + (epsilon_final - epsilon_start) * min(
                1., transitions / max(1., steps * epsilon_decay_fraction))
            if rng.rand() < epsilon:
                actions = rng.randint(0, 5, size=2)
            nxt, features, _ = world.step(state, actions, rng=rng)
            replay.append(state=state, time=t, next_state=nxt,
                          action=int(actions[0] * 5 + actions[1]),
                          done=float(t + 1 == world.horizon), features=features,
                          weights=weights, sources=z)
            state = nxt
            transitions += 1
            if replay.size >= max(warmup, batch_size) and transitions % update_every == 0:
                loss = sf_batch_loss(model, target, replay.sample(rng, batch_size),
                                     world, gamma, sources)
                if not torch.isfinite(loss):
                    raise FloatingPointError("Nonfinite no-context SF loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
                optimizer.step()
                updates += 1
                if updates % target_every == 0:
                    target.load_state_dict(model.state_dict())
            if transitions >= int(steps):
                break
        episodes += 1
        if transitions % 2560 == 0 or transitions >= int(steps):
            progress = {"status": "RUNNING", "version": VERSION,
                        "variant": "no_teammate_conditioning", "steps": transitions,
                        "budget": int(steps), "updates": updates, "episodes": episodes,
                        "test_accessed": False}
            _json_write(out / "progress.json", progress)
            if on_progress is not None:
                on_progress(progress)
    if array_sha256(sources) != source_hash:
        raise RuntimeError("Frozen sources changed during lower training")
    metadata = {"status": "COMPLETE", "version": VERSION,
                "variant": "no_teammate_conditioning", "steps": transitions,
                "budget": int(steps), "updates": updates, "episodes": episodes,
                "seed": int(seed), "gamma": float(gamma), "kappa": float(kappa),
                "horizon": int(horizon), "hidden": int(hidden), "input_dim": 11,
                "source_sha256": source_hash, "initial_model_sha256": initial_hash,
                "model_sha256": model_sha256(model), "code_sha256": file_sha256(__file__),
                "train_tasks_sha256": _task_digest(tasks), "test_accessed": False,
                "training_semantics": "same task-dependent source-profile reference; teammate task omitted from predictor only",
                "recipe": {"lr": lr, "batch_size": batch_size, "warmup": warmup,
                           "update_every": update_every, "target_every": target_every,
                           "epsilon_start": epsilon_start, "epsilon_final": epsilon_final,
                           "epsilon_decay_fraction": epsilon_decay_fraction,
                           "source_temperature": .5, "replay_capacity": 100000}}
    if protocol_dir is not None:
        metadata["protocol_manifest_sha256"] = file_sha256(Path(protocol_dir) / "manifest.json")
        metadata["train_file_sha256"] = file_sha256(Path(protocol_dir) / "train.json")
    payload = dict(metadata)
    payload.update({"model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    "target": {k: v.detach().cpu().clone() for k, v in target.state_dict().items()}})
    _torch_write(out / "lower.pt", payload)
    _json_write(out / "metadata.json", metadata)
    _json_write(out / "progress.json", metadata)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, metadata


def load_ablation_upper(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != VERSION or payload.get("status") != "COMPLETE":
        raise ValueError("Expected a complete component-ablation checkpoint")
    model = make_ablation_model(payload["variant"], gamma=payload["gamma"],
                                horizon=payload["horizon"])
    model.load_state_dict(payload["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload


def load_ablation_lower(path):
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("version") != VERSION:
        return load_context_sf(path)
    if (payload.get("status") != "COMPLETE" or
            payload.get("variant") != "no_teammate_conditioning"):
        raise ValueError("Expected a complete no-teammate lower checkpoint")
    model = NoTeammateContextSF(hidden=payload["hidden"])
    model.load_state_dict(payload["model"])
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model, payload
