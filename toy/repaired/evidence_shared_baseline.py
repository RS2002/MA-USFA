"""Task-conditioned shared joint-action DDQN for the current SFWorld protocol.

This comparison trains one policy across the training objectives at each
coupling/seed.  It has the complete Markov observation and both task vectors,
and can execute all 25 primitive joint actions.  It uses no source policy, SF
predictor, or objective-to-source lookup.  The default optimizer, exploration,
update budget and validation schedule match the attention composer's upper
training; MA-USFA's separate lower-training cost must also be reported.

Scientific training belongs under the external resource guardian.  Importing
this module does not start a run.  Its CLI only loads train and validation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import torch
from torch import nn

from .exact import evaluate_policy
from .learning import StopAtEpisode, model_hash, observation
from .legacy_world import SFWorldExact
from .task_protocol import digest, load_tasks


VERSION = "task-conditioned-shared-joint-ddqn-v1"


@dataclass(frozen=True)
class SharedConfig:
    seed: int = 100
    budget: int = 81920
    horizon: int = 40
    gamma: float = .95
    kappa: float = 1.
    hidden: int = 64
    batch: int = 64
    warmup: int = 256
    update_every: int = 4
    target_every: int = 1000
    replay_capacity: int = 50000
    lr: float = 3e-4
    eval_every: int = 2560
    epsilon_initial: float = .2
    epsilon_final: float = .02
    epsilon_decay_fraction: float = .8
    grad_clip: float = 10.

    def validate(self):
        for field in ("budget", "horizon", "hidden", "batch", "warmup",
                      "update_every", "target_every", "replay_capacity", "eval_every"):
            if int(getattr(self, field)) <= 0:
                raise ValueError(f"{field} must be positive")
        if self.budget % self.horizon or self.eval_every % self.horizon:
            raise ValueError("budget and evaluation interval must align with full episodes")
        if self.batch > self.replay_capacity or self.warmup > self.replay_capacity:
            raise ValueError("replay capacity is too small")
        if self.replay_capacity > 50000:
            raise ValueError("replay capacity exceeds the declared bounded allocation")
        if not 0. <= self.kappa <= 1. or not 0. < self.gamma <= 1.:
            raise ValueError("invalid world configuration")
        if not 0. <= self.epsilon_final <= self.epsilon_initial <= 1.:
            raise ValueError("invalid exploration range")
        if not 0. < self.epsilon_decay_fraction <= 1. or self.lr <= 0.:
            raise ValueError("invalid exploration decay or learning rate")


class SharedJointDDQN(nn.Module):
    """Centralized 13 -> 64 -> 64 -> 25 task-conditioned value network."""

    def __init__(self, hidden=64, horizon=40):
        super().__init__()
        if int(hidden) <= 0 or int(horizon) <= 0:
            raise ValueError("hidden and horizon must be positive")
        self.hidden, self.horizon = int(hidden), int(horizon)
        self.net = nn.Sequential(nn.Linear(13, self.hidden), nn.ReLU(),
                                 nn.Linear(self.hidden, self.hidden), nn.ReLU(),
                                 nn.Linear(self.hidden, 25))

    def spec(self):
        return {"version": VERSION, "hidden": self.hidden,
                "horizon": self.horizon, "observation_dim": 5,
                "task_shape": [2, 4], "joint_actions": 25,
                "architecture": [13, self.hidden, self.hidden, 25],
                "parameters": sum(p.numel() for p in self.parameters()),
                "source_inputs": False, "sf_inputs": False,
                "task_shared": True, "execution": "centralized primitive joint action"}

    def forward(self, obs, weights):
        if obs.ndim != 2 or obs.shape[1] != 5:
            raise ValueError("observation must have shape [B,5]")
        if tuple(weights.shape) != (len(obs), 2, 4):
            raise ValueError("weights must have shape [B,2,4]")
        if obs.device != weights.device or obs.dtype != weights.dtype:
            raise ValueError("observation and weights must share device and dtype")
        if not torch.isfinite(obs).all() or not torch.isfinite(weights).all():
            raise ValueError("observation and weights must be finite")
        q = self.net(torch.cat((obs, weights.reshape(len(obs), 8)), -1))
        if not torch.isfinite(q).all():
            raise FloatingPointError("nonfinite shared DDQN scores")
        return q


class PrimitiveReplay:
    def __init__(self, capacity):
        self.capacity, self.size, self.cursor = int(capacity), 0, 0
        fields = {"state": (np.int64, ()), "time": (np.int16, ()),
                  "next_state": (np.int64, ()), "action": (np.int8, ()),
                  "done": (np.float32, ()), "reward": (np.float32, ()),
                  "weights": (np.float32, (2, 4))}
        self.data = {k: np.zeros((self.capacity,) + shape, dtype=dtype)
                     for k, (dtype, shape) in fields.items()}

    def append(self, **row):
        for key in self.data:
            self.data[key][self.cursor] = row[key]
        self.size = min(self.size + 1, self.capacity)
        self.cursor = (self.cursor + 1) % self.capacity

    def sample(self, rng, batch):
        indices = rng.choice(self.size, int(batch), replace=False)
        return {k: v[indices] for k, v in self.data.items()}


def _inputs(states, times, weights, horizon):
    return (torch.as_tensor(observation(states, times, horizon=horizon)),
            torch.as_tensor(np.array(weights, dtype=np.float32, copy=True)))


def double_dqn_targets(model, target, batch, gamma, horizon):
    """Compute Double-DQN targets, without evaluating terminal observations."""
    rewards = torch.as_tensor(batch["reward"], dtype=torch.float32)
    done = np.asarray(batch["done"])
    if not np.isin(done, (0., 1.)).all():
        raise ValueError("done indicators must be binary")
    alive = done == 0
    result = rewards.clone()
    if alive.any():
        next_times = np.asarray(batch["time"], dtype=np.int64)[alive] + 1
        if np.any(next_times >= horizon):
            raise ValueError("nonterminal transition exceeds finite horizon")
        nx = _inputs(np.asarray(batch["next_state"])[alive], next_times,
                     np.asarray(batch["weights"])[alive], horizon)
        with torch.no_grad():
            greedy = model(*nx).argmax(-1)
            continuation = target(*nx).gather(1, greedy[:, None]).squeeze(1)
        result[torch.as_tensor(alive)] += float(gamma) * continuation
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite Double-DQN target")
    return result.detach()


def shared_ddqn_update(model, target, batch, optimizer, cfg):
    x = _inputs(batch["state"], batch["time"], batch["weights"], cfg.horizon)
    action = torch.as_tensor(batch["action"], dtype=torch.long)
    if ((action < 0) | (action >= 25)).any():
        raise ValueError("primitive joint action outside 0..24")
    prediction = model(*x).gather(1, action[:, None]).squeeze(1)
    expected = double_dqn_targets(model, target, batch, cfg.gamma, cfg.horizon)
    loss = nn.functional.smooth_l1_loss(prediction, expected)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    gradient = nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
    if not torch.isfinite(gradient):
        raise FloatingPointError("nonfinite DDQN gradient")
    optimizer.step()
    return float(loss.detach())


def evaluate_shared_baseline(model, world, tasks):
    """Exact frozen-policy evaluation on the supplied objectives; no updates."""
    if not tasks:
        raise ValueError("evaluation needs at least one objective")
    if model.horizon != world.horizon:
        raise ValueError("checkpoint and world horizons differ")
    before = model_hash(model)
    training = model.training
    model.eval()
    rows = []
    states = np.arange(world.S)
    try:
        with torch.inference_mode():
            for task in tasks:
                weights = np.asarray(task["weights"], dtype=np.float32)
                policy = np.empty((world.horizon, world.S), dtype=np.int64)
                repeated = np.broadcast_to(weights, (world.S, 2, 4))
                for t in range(world.horizon):
                    inputs = _inputs(states, np.full(world.S, t), repeated, world.horizon)
                    policy[t] = model(*inputs).argmax(-1).cpu().numpy()
                rows.append({"task_id": task["id"], "kind": task["kind"],
                             "return": evaluate_policy(world, policy, weights)["initial_return"]})
    finally:
        model.train(training)
    after = model_hash(model)
    if before != after:
        raise RuntimeError("frozen evaluation changed model parameters")
    groups = sorted({r["kind"] for r in rows})
    return {"mean_return": float(np.mean([r["return"] for r in rows])),
            "group_means": {g: float(np.mean([r["return"] for r in rows
                                               if r["kind"] == g])) for g in groups},
            "rows": rows, "optimizer_updates": 0,
            "model_sha256_before": before, "model_sha256_after": after,
            "evaluation": "exact finite-horizon expectation over declared initial distribution"}


# Kept as a short public alias for the common frozen-evaluation interface.
evaluate_shared = evaluate_shared_baseline


def _json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _checkpoint(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    os.replace(temporary, path)


def _file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _code_hashes():
    root = Path(__file__).resolve().parent
    return {str(root / name): _file_hash(root / name) for name in
            ("evidence_shared_baseline.py", "learning.py", "legacy_world.py",
             "exact.py", "task_protocol.py")}


def _check_code(hashes):
    if any(_file_hash(path) != expected for path, expected in hashes.items()):
        raise RuntimeError("training implementation changed during execution")


def _validate_tasks(tasks, name):
    if not tasks:
        raise ValueError(f"{name} objectives cannot be empty")
    ids = []
    for task in tasks:
        if not task.get("id") or not task.get("kind"):
            raise ValueError(f"{name} objective lacks id/kind")
        weights = np.asarray(task["weights"], dtype=np.float32)
        if weights.shape != (2, 4) or not np.isfinite(weights).all():
            raise ValueError(f"invalid {name} task weights")
        ids.append(task["id"])
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate {name} objective ids")


def train_shared_baseline(out_dir, world, train_tasks, val_tasks, *, seed=100,
                          budget=81920, hidden=64, batch=64, warmup=256,
                          update_every=4, target_every=1000, lr=3e-4,
                          eval_every=2560, replay_capacity=50000,
                          protocol_hashes=None):
    """Train one shared model and retain best-validation and endpoint separately.

    Existing run folders are rejected.  Interrupted runs retain their records
    and are never silently restarted or treated as complete.
    """
    threads = int(os.environ.get("MA_USFA_THREAD_BUDGET", "0"))
    if threads not in (1, 2):
        raise RuntimeError("run through a resource guardian with a one/two-thread budget")
    cfg = SharedConfig(seed=seed, budget=budget, horizon=world.horizon,
                       gamma=world.gamma, kappa=world.kappa, hidden=hidden,
                       batch=batch, warmup=warmup, update_every=update_every,
                       target_every=target_every, replay_capacity=replay_capacity,
                       lr=lr, eval_every=eval_every)
    cfg.validate()
    _validate_tasks(train_tasks, "training")
    _validate_tasks(val_tasks, "validation")
    if {t["id"] for t in train_tasks} & {t["id"] for t in val_tasks}:
        raise ValueError("training and validation objective ids overlap")
    train_weight_set = {np.asarray(t["weights"], dtype=np.float32).tobytes() for t in train_tasks}
    if any(np.asarray(t["weights"], dtype=np.float32).tobytes() in train_weight_set for t in val_tasks):
        raise ValueError("training and validation contain identical objectives")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if any(out.iterdir()):
        raise FileExistsError("output run directory is not empty; completed/failed runs are immutable")
    torch.set_num_threads(threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(seed)
    rng = np.random.RandomState(seed + 17)
    model = SharedJointDDQN(hidden, horizon=world.horizon)
    target = SharedJointDDQN(hidden, horizon=world.horizon)
    target.load_state_dict(model.state_dict())
    for p in target.parameters():
        p.requires_grad_(False)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    replay = PrimitiveReplay(replay_capacity)
    code_hashes = _code_hashes()
    input_hashes = {"train_objectives": digest(train_tasks),
                    "validation_objectives": digest(val_tasks),
                    **(protocol_hashes or {})}
    identity = {"version": VERSION, "config": asdict(cfg), "model_spec": model.spec(),
                "protocol_hashes": input_hashes, "code_sha256": code_hashes,
                "train_task_ids": [t["id"] for t in train_tasks],
                "validation_task_ids": [t["id"] for t in val_tasks],
                "test_objectives_loaded": False, "device": "cpu", "threads": threads,
                "initial_parameter_sha256": model_hash(model),
                "cost_scope": "shared-policy transitions; report MA-USFA lower pretraining separately"}
    _json(out / "recipe.json", identity)
    _json(out / "status.json", {"status": "RUNNING", "steps": 0, "test_objectives_loaded": False})
    logs = []
    steps = updates = episodes = 0
    best = -float("inf")
    best_step = None
    loss = None
    start = time.monotonic()

    def payload(selection, validation):
        return {**identity, "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "selection": selection, "steps": steps, "updates": updates,
                "episodes": episodes, "validation": validation,
                "parameter_sha256": model_hash(model)}

    try:
        with StopAtEpisode() as stop:
            while steps < budget:
                task = train_tasks[int(rng.randint(len(train_tasks)))]
                weights = np.asarray(task["weights"], dtype=np.float32)
                state = world.reset(rng=rng)
                for t in range(world.horizon):
                    epsilon = max(cfg.epsilon_final, cfg.epsilon_initial *
                                  (1. - steps / max(1., budget * cfg.epsilon_decay_fraction)))
                    with torch.no_grad():
                        q = model(*_inputs(np.array([state]), np.array([t]), weights[None], world.horizon))[0]
                    action = int(rng.randint(25)) if rng.rand() < epsilon else int(q.argmax())
                    nxt, features, _ = world.step(state, action, rng=rng)
                    replay.append(state=state, time=t, next_state=nxt, action=action,
                                  reward=float(world.reward(features, weights)),
                                  done=float(t + 1 == world.horizon), weights=weights)
                    state = nxt
                    steps += 1
                    if replay.size >= max(warmup, batch) and steps % update_every == 0:
                        loss = shared_ddqn_update(model, target, replay.sample(rng, batch), optimizer, cfg)
                        updates += 1
                        if updates % target_every == 0:
                            target.load_state_dict(model.state_dict())
                episodes += 1
                if steps % eval_every == 0 or steps == budget:
                    _check_code(code_hashes)
                    validation = evaluate_shared_baseline(model, world, val_tasks)
                    log = {"steps": steps, "updates": updates, "episodes": episodes,
                           "epsilon": epsilon, "loss": loss, "validation": validation,
                           "elapsed_seconds": time.monotonic() - start}
                    logs.append(log)
                    with (out / "progress.jsonl").open("a") as f:
                        f.write(json.dumps(log, allow_nan=False) + "\n")
                    if validation["mean_return"] > best:
                        best, best_step = validation["mean_return"], steps
                        _checkpoint(out / "best.pt", payload("best validation mean over all objectives", validation))
                    _json(out / "status.json", {"status": "RUNNING", "steps": steps,
                                                "updates": updates, "best_step": best_step,
                                                "test_objectives_loaded": False})
                    print(json.dumps({"method": VERSION, "steps": steps, "updates": updates,
                                      "validation_mean": validation["mean_return"],
                                      "elapsed_seconds": log["elapsed_seconds"]}), flush=True)
                if stop.requested and steps < budget:
                    _checkpoint(out / "interrupted.pt", payload("interrupted; not a complete run", None))
                    _json(out / "status.json", {"status": "INTERRUPTED", "steps": steps,
                                                "updates": updates, "complete": False,
                                                "test_objectives_loaded": False})
                    return None, {"status": "INTERRUPTED", "steps": steps, "updates": updates,
                                  "complete": False, "test_objectives_loaded": False}
        _check_code(code_hashes)
        endpoint_validation = logs[-1]["validation"]
        _checkpoint(out / "endpoint.pt", payload("fixed-budget endpoint", endpoint_validation))
        summary = {**identity, "status": "PASS", "complete": True, "steps": steps,
                   "updates": updates, "episodes": episodes, "best_validation": best,
                   "best_step": best_step, "endpoint_validation": endpoint_validation,
                   "best_checkpoint": "best.pt", "endpoint_checkpoint": "endpoint.pt",
                   "logs": logs, "elapsed_seconds": time.monotonic() - start}
        _json(out / "summary.json", summary)
        _json(out / "status.json", {"status": "PASS", "steps": steps, "updates": updates,
                                    "complete": True, "test_objectives_loaded": False})
        best_model, _ = load_shared_baseline(out / "best.pt")
        return best_model, summary
    except BaseException as exc:
        _json(out / "status.json", {"status": "FAILED", "steps": steps, "updates": updates,
                                    "error": str(exc), "complete": False,
                                    "test_objectives_loaded": False})
        raise


def load_shared_baseline(path):
    """Load a CPU checkpoint and verify its architecture and parameter identity."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    if payload.get("version") != VERSION:
        raise ValueError("unexpected shared DDQN checkpoint version")
    config = payload["config"]
    model = SharedJointDDQN(config["hidden"], config["horizon"])
    if model.spec() != payload["model_spec"]:
        raise ValueError("shared DDQN architecture metadata differs")
    model.load_state_dict(payload["model"], strict=True)
    if model_hash(model) != payload.get("parameter_sha256"):
        raise ValueError("shared DDQN parameter hash differs")
    if any(not torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("checkpoint has nonfinite parameters")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    return model, payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=100)
    parser.add_argument("--kappa", type=float, required=True)
    parser.add_argument("--budget", type=int, default=81920)
    args = parser.parse_args()
    protocol = args.protocol_dir.resolve()
    train = load_tasks(protocol, "train")
    validation = load_tasks(protocol, "validation")
    if len(train) != 64 or len(validation) != 16:
        raise ValueError("current scientific protocol requires 64 train and 16 validation objectives")
    hashes = {name: _file_hash(protocol / name) for name in ("manifest.json", "train.json", "validation.json")}
    world = SFWorldExact(horizon=40, gamma=.95, kappa=args.kappa)
    _, summary = train_shared_baseline(args.output_dir, world, train, validation,
                                       seed=args.seed, budget=args.budget, protocol_hashes=hashes)
    print(json.dumps({k: summary[k] for k in ("status", "steps", "updates", "complete")}, indent=2))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
