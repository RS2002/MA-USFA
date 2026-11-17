"""Paired upper-layer retraining with only mixer normalization varied.

The training loop and schedules mirror tsc_repaired.train_online_upper.
Both arms start from the same random seed, with frozen source/SF inputs.
No validation observations participate in fitting or endpoint selection.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import random
import time

import numpy as np
import torch

from tsc_repaired.config import load_manifest
from tsc_repaired.task_selector import (
    TaskDDQNTrainer, TaskSelectorConfig, TaskTransition, bounded_replay_capacity,
)
from tsc_repaired.train_online_upper import (
    sha256, _source_hashes, _numpy_inputs, _tensor_inputs,
)
from .selector import (
    VARIANTS, checkpoint_spec, parameter_sha256,
    selector_for_variant, version_for_variant,
)


def train_ablation_upper(manifest_path, source_dir, sf_checkpoint, output, variant,
                       seed=100, episodes=10, updates_per_step=1, batch_size=16,
                       replay_capacity=1024, device="cpu", guard=None) -> dict:
    # Import at call time so the selector's mathematical tests do not need a
    # SUMO installation and can independently validate DDQN and topology.
    from tsc_repaired.traffic_runtime import FrozenTrafficRuntime
    from tsc_repaired.task_protocol_v2 import load_protocol

    if variant not in VARIANTS:
        raise ValueError("variant must be control or normalized")
    manifest = load_manifest(manifest_path)
    manifest.require_stage("pilot")
    if int(episodes) <= 0 or int(updates_per_step) <= 0 or int(batch_size) <= 0:
        raise ValueError("episodes, updates_per_step and batch_size must be positive")
    if int(batch_size) > 16 or int(replay_capacity) > 1024:
        raise ValueError("development trainer is limited to batch_size<=16 and replay_capacity<=1024")
    output = Path(output).expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "upper_ddqn_v2.pt"
    if checkpoint.exists() or (output / "train_upper_v2.json").exists():
        raise FileExistsError("upper checkpoint already exists; use a new run directory")
    sf_checkpoint = Path(sf_checkpoint).expanduser().resolve()
    if not sf_checkpoint.is_file():
        raise FileNotFoundError(sf_checkpoint)
    source_hashes_before = _source_hashes(source_dir)
    sf_hash_before = sha256(sf_checkpoint)
    random.seed(int(seed)); np.random.seed(int(seed)); torch.manual_seed(int(seed))
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(int(seed))
    rng = np.random.default_rng(int(seed))
    started = time.monotonic()
    if guard is not None:
        guard.check()
    metrics, episode_reports = [], []
    transitions, trainer, config, protocol = 0, None, None, None
    replay_bytes_per_row = effective_capacity = None
    initial_parameter_hash = None
    runtime = FrozenTrafficRuntime(manifest_path, source_dir, sf_checkpoint=sf_checkpoint, device=device)
    try:
        protocol = load_protocol(manifest_path, runtime.ids)
        tasks = protocol.tasks("train")
        traffic_seeds = protocol.traffic_seeds("train")
        if not tasks or not traffic_seeds:
            raise ValueError("training task/traffic seed split is empty")
        if dict(runtime.source_hashes) != source_hashes_before:
            raise ValueError("runtime source identity differs from the checkpoint files")
        adjacency = np.asarray(runtime.adjacency, dtype=bool)
        for episode in range(int(episodes)):
            task = tasks[episode % len(tasks)]
            traffic_seed = int(traffic_seeds[episode % len(traffic_seeds)])
            state = runtime.reset(task, traffic_seed)
            if guard is not None:
                guard.check()
            current = _numpy_inputs(runtime, state, 0)
            if config is None:
                # The lower's discount fixes the scale/continuation semantics.
                payload = torch.load(sf_checkpoint, map_location="cpu", weights_only=False)
                gamma = float(payload["config"]["gamma"])
                config = TaskSelectorConfig(
                    observation_dim=runtime.observation_dim, source_count=runtime.K,
                    action_count=runtime.action_count, task_dim=int(runtime.weights.shape[-1]),
                    gamma=gamma, horizon=runtime.horizon)
                online = selector_for_variant(config, variant).to(device).set_adjacency(adjacency)
                initial_parameter_hash = parameter_sha256(online)
            episode_team_return = 0.
            episode_steps = 0
            selected_counts = np.zeros(runtime.K, dtype=np.int64)
            episode_metric_start = len(metrics)
            mixing_means, mixing_mins, mixing_maxes = [], [], []
            # Exploration depends only on elapsed training steps; no validation
            # target or held-out reward participates in the schedule.
            planned_steps = max(1, int(episodes) * runtime.horizon)
            for step in range(runtime.horizon):
                if guard is not None:
                    guard.check()
                model = trainer.online if trainer is not None else online
                fraction = transitions / max(1., planned_steps * .8)
                epsilon = .30 + (.05 - .30) * min(1., fraction)
                with torch.no_grad():
                    parts = model.components(*_tensor_inputs(current, device))
                    entries = parts["utilities"].argmax(-1)[0].cpu().numpy()
                    mixing_means.append(float(parts["mixing"].mean().cpu()))
                    mixing_mins.append(float(parts["mixing"].min().cpu()))
                    mixing_maxes.append(float(parts["mixing"].max().cpu()))
                entries = np.asarray(entries, dtype=np.int64)
                explore = rng.random(runtime.N) < epsilon
                entries[explore] = rng.integers(0, runtime.K, size=int(explore.sum()))
                result = runtime.step(entries)
                done = bool(result.done or step + 1 >= runtime.horizon)
                reward_values = np.asarray(result.rewards, dtype=np.float64)
                if reward_values.shape != (runtime.N,) or not np.isfinite(reward_values).all():
                    raise ValueError("runtime must return finite rewards for every controlled node")
                episode_team_return += float(reward_values.sum())
                episode_steps += 1
                selected_counts += np.bincount(entries, minlength=runtime.K)
                next_step = step + 1
                if done:
                    # Terminal rows are never bootstrapped. Retain shapes and a
                    # finite terminal observation without querying out-of-range
                    # source time or creating unnecessary frozen SF work.
                    next_inputs = (np.array(result.state, dtype=np.float32, copy=True),
                                   current[1].copy(), np.zeros_like(current[2]),
                                   np.zeros_like(current[3]), next_step / max(1, runtime.horizon))
                else:
                    next_inputs = _numpy_inputs(runtime, result.state, next_step)
                row = TaskTransition(
                    obs=current[0], weights=current[1], candidate_actions=current[2],
                    sf_values=current[3], time_fraction=current[4], entries=entries.copy(),
                    reward=float(reward_values.mean()), next_obs=next_inputs[0],
                    next_candidate_actions=next_inputs[2], next_sf_values=next_inputs[3],
                    next_time_fraction=next_inputs[4], done=done)
                if trainer is None:
                    max_rss = float(manifest.resource_limits["max_rss_gib"]) * 1024**3
                    max_replay_bytes = int(min(256 * 1024**2, .10 * max_rss))
                    effective_capacity, replay_bytes_per_row = bounded_replay_capacity(
                        int(replay_capacity), row, max_replay_bytes, int(batch_size))
                    trainer = TaskDDQNTrainer(online, seed=seed, replay_capacity=effective_capacity,
                                              target_every=100, device=device)
                trainer.add(row)
                transitions += 1
                if len(trainer.replay) >= int(batch_size):
                    for _ in range(int(updates_per_step)):
                        if guard is not None:
                            guard.check()
                        metrics.append(trainer.update(batch_size=int(batch_size)))
                current = next_inputs
                if done:
                    break
            episode_reports.append({
                "episode": episode, "task_id": str(task["id"]), "traffic_seed": traffic_seed,
                "steps": episode_steps, "episode_team_return": episode_team_return,
                "episode_mean_agent_return": episode_team_return / runtime.N,
                "mean_step_node_reward": episode_team_return / max(1, runtime.N * episode_steps),
                "source_selection_histogram": {str(i): int(v) for i, v in enumerate(selected_counts)},
                "epsilon_end": float(epsilon), "optimizer_updates": trainer.updates,
                "mean_training_loss": (float(np.mean([m["loss"] for m in metrics[episode_metric_start:]]))
                                       if len(metrics) > episode_metric_start else None),
                "observed_mixing_mean": float(np.mean(mixing_means)),
                "observed_mixing_min": float(min(mixing_mins)),
                "observed_mixing_max": float(max(mixing_maxes)),
            })
        if trainer is None or not metrics:
            raise RuntimeError("training produced no optimizer updates; increase episodes or reduce batch size")
        if _source_hashes(source_dir) != source_hashes_before or sha256(sf_checkpoint) != sf_hash_before:
            raise RuntimeError("a frozen source/SF checkpoint changed during upper training")
        frozen_audit = runtime.audit_frozen()
        metadata = {
            "ablation_variant": variant,
            "initial_parameter_sha256": initial_parameter_hash,
            "training_configuration": {
                "updates_per_step": int(updates_per_step), "batch_size": int(batch_size),
                "requested_replay_capacity": int(replay_capacity), "learning_rate": 3e-4,
                "target_every": 100, "epsilon_start": .30, "epsilon_end": .05,
                "epsilon_decay_fraction_of_planned_steps": .8,
            },
            "source_hashes": source_hashes_before, "sf_sha256": sf_hash_before,
            "protocol_fingerprint": protocol.fingerprint, "node_ids": list(runtime.ids),
            "training_split": "train", "seed": int(seed), "episodes": int(episodes),
            "transitions": transitions, "optimizer_updates": trainer.updates,
            "checkpoint_selection": "fixed-budget endpoint; no validation selection",
            "test_accessed": False, "test_updates": 0,
            "training_reward_reduction": "mean node reward",
        }
        saved = {
            "version": version_for_variant(variant), "config": asdict(config),
            "online": {k: v.detach().cpu() for k, v in trainer.online.state_dict().items()},
            "target": {k: v.detach().cpu() for k, v in trainer.target.state_dict().items()},
            "optimizer": trainer.optimizer.state_dict(), "metadata": metadata,
            "adjacency": adjacency.tolist(), "spec": checkpoint_spec(trainer.online, variant),
        }
        temp_checkpoint = checkpoint.with_name(checkpoint.name + ".tmp")
        torch.save(saved, temp_checkpoint)
        temp_checkpoint.replace(checkpoint)
        report = {
            "status": "PASS", "version": version_for_variant(variant), "algorithm": "team DDQN",
            "ablation_variant": variant, "mixing_normalization": "none" if variant == "control" else "mean_one",
            "initial_parameter_sha256": initial_parameter_hash,
            "training_configuration": metadata["training_configuration"],
            "collection": "online SUMO control under the current epsilon-greedy selector",
            "training_split": "train", "seed": int(seed), "episodes": int(episodes),
            "transitions": transitions, "optimizer_updates": trainer.updates,
            "test_accessed": False, "test_updates": 0,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256(checkpoint),
            "checkpoint_selection": metadata["checkpoint_selection"],
            "source_hashes": source_hashes_before, "sf_sha256": sf_hash_before,
            "protocol_fingerprint": protocol.fingerprint,
            "replay_capacity": effective_capacity, "replay_estimated_bytes_per_row": replay_bytes_per_row,
            "last_metrics": metrics[-1], "episode_reports": episode_reports,
            "elapsed_seconds": time.monotonic() - started,
            "policy_validation_evaluated": False, "frozen_source_sf_audit": frozen_audit,
        }
        (output / "train_upper_v2.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        return report
    finally:
        runtime.close()

