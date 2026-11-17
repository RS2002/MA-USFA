"""Train scene-bound sources with the paper's per-node source-DQN recipe.

This source pretraining is separate from the upper team-DDQN objective.
Each source fits local node rewards with a shared per-node Q network and
copies its target every 100 episodes. Terminal/error/provenance checks do
not change that objective or copy schedule.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import load_manifest, sha256_file
from .features import FEATURE_NAMES, TrafficFeatureExtractor
from .scenario import ScenarioSpec, inspect_network
from .source_policy import SourceQNet, observation_array
from .task_protocol_v2 import load_protocol


TRAINING_VERSION = "ma-usfa-traffic-source-training-local-dqn-bound-v1"


def _check(guard) -> None:
    if guard is not None:
        guard.check()


def _reset(env, seed: int):
    result = env.reset(seed=int(seed))
    return result[0] if isinstance(result, tuple) and len(result) == 2 else result


def _step(env, actions: dict[str, int], ids: tuple[str, ...]):
    result = env.step(actions)
    if len(result) == 5:
        state, rewards, terminated, truncated, info = result
        if not isinstance(terminated, dict) or not isinstance(truncated, dict):
            raise ValueError("parallel SUMO termination values must be dictionaries")
        done = all(bool(terminated.get(node, False) or truncated.get(node, False)) for node in ids)
    elif len(result) == 4:
        state, rewards, done, info = result
        if isinstance(done, dict):
            done = all(bool(done.get(node, False)) for node in ids)
        else:
            done = bool(done)
    else:
        raise RuntimeError("unsupported parallel_env.step signature")
    if not isinstance(rewards, dict) or any(node not in rewards for node in ids):
        raise ValueError("SUMO reward is missing a controlled node")
    values = np.asarray([float(rewards[node]) for node in ids], dtype=np.float32)
    if not np.isfinite(values).all():
        raise FloatingPointError("source training received non-finite node rewards")
    return state, values, bool(done), info


def _action_count(env, ids: tuple[str, ...]) -> int:
    counts = []
    for node in ids:
        spaces = getattr(env, "action_spaces", None)
        space = spaces[node] if spaces is not None else env.action_space(node)
        if not hasattr(space, "n") or isinstance(space.n, bool) or int(space.n) != space.n or int(space.n) <= 0:
            raise ValueError("source training requires positive discrete primitive action spaces")
        counts.append(int(space.n))
    if len(set(counts)) != 1:
        raise ValueError("controlled nodes have different primitive action counts")
    return counts[0]


def _dqn_target(target, next_state: torch.Tensor,
                next_time: torch.Tensor, rewards,
                done: bool, gamma: float) -> torch.Tensor:
    """Per-node vanilla DQN targets; terminal rows never bootstrap."""
    result = torch.as_tensor(rewards, dtype=torch.float32, device=next_state.device)
    if result.ndim != 1 or result.shape[0] != next_state.shape[1]:
        raise ValueError("source reward target must have one entry per node")
    if not done:
        with torch.no_grad():
            next_q = target(next_state, next_time)[0].max(dim=-1).values
            result = result + float(gamma) * next_q
    if not bool(torch.isfinite(result).all()):
        raise FloatingPointError("source DQN target is non-finite")
    return result


def _settings(manifest) -> dict[str, Any]:
    raw = manifest.raw.get("source_training", {})
    if not isinstance(raw, dict):
        raise ValueError("source_training must be an object")
    result = {"hidden": 128, "lr": 1e-3, "gamma": 0.99, "target_every_episodes": 100,
              "epsilon_start": 1.0, "epsilon_final": 0.05, "grad_clip": 10.0}
    result.update({key: raw[key] for key in result if key in raw})
    for name in ("hidden", "target_every_episodes"):
        value = result[name]
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"source_training.{name} must be a positive integer")
    for name in ("lr", "grad_clip"):
        value = result[name]
        if isinstance(value, bool) or not math.isfinite(float(value)) or float(value) <= 0:
            raise ValueError(f"source_training.{name} must be positive and finite")
    for name in ("gamma", "epsilon_start", "epsilon_final"):
        value = result[name]
        if isinstance(value, bool) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise ValueError(f"source_training.{name} must lie in [0,1]")
    return result


def train_sources_v2(manifest_path, output, seed=0, episodes=5,
                     device="cpu", guard=None) -> dict:
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("source seed must be a nonnegative integer")
    if isinstance(episodes, bool) or not isinstance(episodes, int) or episodes <= 0:
        raise ValueError("source episodes must be a positive integer")
    manifest = load_manifest(manifest_path)
    manifest.require_stage("pilot")
    spec = ScenarioSpec.from_manifest(manifest)
    ids = inspect_network(spec.files.network).controlled_signal_ids
    if not ids:
        raise ValueError("network declares no controlled traffic signals")
    protocol = load_protocol(manifest_path, node_ids=ids)
    traffic_seeds = protocol.traffic_seeds("train")
    if not traffic_seeds:
        raise ValueError("source training has no declared training traffic seeds")
    settings = _settings(manifest)
    scenario_hashes = spec.files.hashes()
    scenario_settings = {key: getattr(spec, key) for key in (
        "delta_time", "horizon_steps", "yellow_time", "min_green", "max_green", "scale")}
    out = Path(output).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    if list(out.glob("source_*.pt")) or list(out.glob("source_*.json")) or (out / "train_sources_v2.json").exists():
        raise FileExistsError("source output already contains results; use a fresh run directory")
    reports = []
    shared_dimensions = None
    _check(guard)
    for source_index, anchor in enumerate(protocol.anchors):
        # This seed is used before either network is constructed.
        model_seed = int(seed) + int(source_index)
        torch.manual_seed(model_seed)
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(model_seed)
        online = target = optimizer = None
        updates = 0
        target_copies = 0
        training_returns = []
        team_returns = []
        episode_seeds = []
        for episode in range(episodes):
            _check(guard)
            # The original recipe offsets the episode seed by source index.
            # Cycle only within the predeclared training seed set.
            traffic_seed = int(traffic_seeds[(model_seed + episode) % len(traffic_seeds)])
            episode_seeds.append(traffic_seed)
            extractor = TrafficFeatureExtractor()

            def reward_fn(signal, _extractor=extractor, _weights=anchor):
                return _extractor.reward(signal, weights=_weights)

            env = None
            try:
                env = spec.build_parallel_env(reward_fn, sumo_seed=traffic_seed)
                actual_ids = tuple(sorted(str(node) for node in getattr(
                    env, "possible_agents", getattr(env, "agents", ()))))
                if actual_ids != ids:
                    raise ValueError("SUMO controlled nodes differ from the declared source node order")
                observation = _reset(env, traffic_seed)
                state = observation_array(observation, ids)
                action_count = _action_count(env, ids)
                dimensions = (int(state.shape[-1]), action_count)
                if shared_dimensions is None:
                    shared_dimensions = dimensions
                if dimensions != shared_dimensions:
                    raise ValueError("source observation/action dimensions changed across environments")
                if online is None:
                    online = SourceQNet(dimensions[0], dimensions[1], settings["hidden"]).to(device)
                    target = SourceQNet(dimensions[0], dimensions[1], settings["hidden"]).to(device)
                    target.load_state_dict(online.state_dict())
                    target.eval()
                    target.requires_grad_(False)
                    optimizer = torch.optim.Adam(online.parameters(), lr=float(settings["lr"]))
                episode_return = 0.0
                team_return = 0.0
                epsilon = float(settings["epsilon_start"]) + (
                    float(settings["epsilon_final"]) - float(settings["epsilon_start"])) * (
                    episode / max(1, episodes - 1))
                for step in range(spec.horizon_steps):
                    _check(guard)
                    state_t = torch.as_tensor(state[None], dtype=torch.float32, device=device)
                    time_t = torch.tensor([step / spec.horizon_steps], dtype=torch.float32, device=device)
                    with torch.no_grad():
                        greedy = online(state_t, time_t)[0].argmax(-1).cpu().numpy()
                    actions = greedy.astype(np.int64, copy=True)
                    mask = np.random.default_rng(model_seed + episode * 1000 + step).random(len(ids)) < epsilon
                    exploration = np.random.default_rng(model_seed + episode * 1000 + step + 1).integers(
                        0, action_count, size=len(ids))
                    actions[mask] = exploration[mask]
                    next_obs, reward_vec, env_done, _ = _step(
                        env, {node: int(action) for node, action in zip(ids, actions)}, ids)
                    # The explicit finite-horizon cap is a terminal transition,
                    # even when an environment wrapper does not mark it so.
                    done = bool(env_done or step + 1 >= spec.horizon_steps)
                    if next_obs:
                        next_state = observation_array(next_obs, ids)
                        if next_state.shape != state.shape:
                            raise ValueError("source observation shape changed during an episode")
                    elif done:
                        next_state = np.zeros_like(state)
                    else:
                        raise ValueError("nonterminal SUMO step omitted source observations")
                    next_t = torch.tensor([(step + 1) / spec.horizon_steps], dtype=torch.float32, device=device)
                    next_state_t = torch.as_tensor(next_state[None], dtype=torch.float32, device=device)
                    q = online(state_t, time_t)[0]
                    chosen = q.gather(-1, torch.as_tensor(actions, dtype=torch.long, device=device)[:, None]).squeeze(-1)
                    reward = float(reward_vec.astype(np.float64).mean())
                    y = _dqn_target(target, next_state_t, next_t, reward_vec, done, settings["gamma"])
                    loss = torch.nn.functional.smooth_l1_loss(chosen, y)
                    if not bool(torch.isfinite(loss)):
                        raise FloatingPointError("source training loss is non-finite")
                    _check(guard)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(online.parameters(), float(settings["grad_clip"]))
                    if not bool(torch.isfinite(grad_norm)):
                        raise FloatingPointError("source training gradient is non-finite")
                    optimizer.step()
                    if any(not bool(torch.isfinite(parameter).all()) for parameter in online.parameters()):
                        raise FloatingPointError("source optimizer produced non-finite parameters")
                    updates += 1
                    episode_return += reward
                    team_return += float(reward_vec.astype(np.float64).sum())
                    state = next_state
                    if done:
                        break
                training_returns.append(episode_return)
                team_returns.append(team_return)
                if (episode + 1) % int(settings["target_every_episodes"]) == 0:
                    target.load_state_dict(online.state_dict())
                    target_copies += 1
            finally:
                if env is not None:
                    env.close()
        _check(guard)
        if spec.files.hashes() != scenario_hashes or sha256_file(protocol.path) != protocol.fingerprint:
            raise RuntimeError("scenario or task protocol bytes changed during source training")
        checkpoint = out / f"source_{source_index:03d}.pt"
        payload = {
            "state_dict": {name: value.detach().cpu().clone() for name, value in online.state_dict().items()},
            "source_index": int(source_index), "observation_dim": shared_dimensions[0],
            "action_count": shared_dimensions[1], "hidden": int(settings["hidden"]),
            "weights": np.asarray(anchor, dtype=np.float32).tolist(), "seed": model_seed,
            "training_seed": int(seed), "episodes": episodes, "updates": updates,
            "scenario_hashes": scenario_hashes, "scenario_settings": scenario_settings,
            "protocol_fingerprint": protocol.fingerprint,
            "protocol_sha256": protocol.fingerprint, "node_ids": list(ids),
            "feature_names": list(FEATURE_NAMES), "training_version": TRAINING_VERSION,
            "source_training_traffic_seeds": episode_seeds, "source_training": settings,
            "training_reward_reduction": "per_node_rewards_in_per_node_DQN_targets",
            "q_reduction": "mean_Huber_loss_over_nodes",
            "algorithm": "one_step_per_node_dqn", "test_updates": 0,
            "target_copies": target_copies, "target_copy_unit": "episodes",
            "validation_objectives_used_for_training": False,
            "test_objectives_loaded": False,
        }
        # Exclusive creation preserves all previously saved source artifacts.
        with checkpoint.open("xb") as stream:
            torch.save(payload, stream)
        report = {
            "status": "PASS", "source_index": int(source_index),
            "weights": payload["weights"], "seed": model_seed,
            "episodes": episodes, "updates": updates,
            "mean_training_episode_mean_agent_return": float(np.mean(training_returns)),
            "mean_training_episode_team_return": float(np.mean(team_returns)),
            "source_training_traffic_seeds": episode_seeds,
            "checkpoint": str(checkpoint), "checkpoint_sha256": sha256_file(checkpoint),
            "scenario_hashes": scenario_hashes,
            "scenario_settings": scenario_settings,
            "target_copies": target_copies, "target_copy_unit": "episodes",
            "protocol_fingerprint": protocol.fingerprint, "training_version": TRAINING_VERSION,
            "test_updates": 0,
        }
        with checkpoint.with_suffix(".json").open("x", encoding="utf-8") as stream:
            json.dump(report, stream, indent=2)
            stream.write("\n")
        reports.append(report)
    result = {
        "status": "PASS", "training_version": TRAINING_VERSION,
        "algorithm": "one_step_per_node_dqn", "sources": reports,
        "source_hashes": {Path(row["checkpoint"]).name: row["checkpoint_sha256"] for row in reports},
        "source_episodes": episodes, "source_count": len(reports),
        "scenario_hashes": scenario_hashes, "scenario_settings": scenario_settings,
        "protocol_fingerprint": protocol.fingerprint,
        "training_seed": int(seed), "node_ids": list(ids),
        "source_training": settings,
        "test_updates": 0, "test_objectives_loaded": False,
        "validation_objectives_used_for_training": False,
    }
    with (out / "train_sources_v2.json").open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    try:
        result = train_sources_v2(args.manifest, args.output, args.seed,
                                  args.episodes, args.device)
        print(json.dumps(result, indent=2))
        return 0
    except (OSError, ValueError, RuntimeError, ImportError, FloatingPointError) as exc:
        print(json.dumps({"status": "NOT_READY", "error": str(exc),
                          "test_objectives_loaded": False}, indent=2))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
