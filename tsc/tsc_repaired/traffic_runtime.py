"""Frozen-source traffic execution shared by collection and closed-loop evaluation.

Task weights are set before an episode is constructed.  SF candidate values
depend on those declared weights and current observations, never on the
source indices which a selector is about to choose.  SUMO is imported only
when ``reset`` constructs an explicitly seeded environment.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from .config import load_manifest, sha256_file
from .features import FEATURE_NAMES, TrafficFeatureExtractor
from .scenario import ScenarioSpec, inspect_network
from .source_library import FrozenSourcePolicyLibrary
from .source_policy import SourceCheckpoint, observation_array
from .task_protocol_v2 import load_protocol, validate_task_weights


@dataclass(frozen=True)
class StepResult:
    state: np.ndarray
    rewards: np.ndarray
    done: bool
    info: Any
    features: np.ndarray
    actions: np.ndarray


def _torch_load(path: Path, device: str):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # Older PyTorch servers.
        return torch.load(path, map_location=device)


def _model_hash(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        array = value.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


class FrozenTrafficRuntime:
    """Execute only primitive actions supplied by frozen source policies.

    Source files, task protocol and optional SF are checked before SUMO is
    started.  ``close`` closes only the current environment, so the same
    immutable models can be reset for another paired traffic seed.
    """

    def __init__(self, manifest_path: str | Path, source_dir: str | Path,
                 sf_checkpoint: str | Path | None = None, device: str = "cpu"):
        self.manifest = load_manifest(manifest_path)
        self.manifest.require_stage("pilot")
        self.device = str(device)
        self.spec = ScenarioSpec.from_manifest(self.manifest)
        self.manifest_hash = sha256_file(self.manifest.path)
        self.scenario_settings = {
            name: getattr(self.spec, name) for name in (
                "delta_time", "horizon_steps", "yellow_time", "min_green",
                "max_green", "scale")
        }
        self.ids = inspect_network(self.spec.files.network).controlled_signal_ids
        if not self.ids:
            raise ValueError("declared network contains no controlled signals")
        self.N = len(self.ids)
        self.horizon = int(self.spec.horizon_steps)
        self.protocol = load_protocol(manifest_path, node_ids=self.ids)
        self.anchors = np.asarray(self.protocol.anchors, dtype=np.float32).copy()
        self.K = int(len(self.anchors))
        if self.K <= 0:
            raise ValueError("task protocol declares no source anchors")
        self.source_dir = Path(source_dir).expanduser().resolve()
        paths = sorted(self.source_dir.glob("source_*.pt"))
        if len(paths) != self.K:
            raise ValueError(f"expected {self.K} source checkpoints, found {len(paths)}")
        self.source_paths = tuple(paths)
        self.source_hashes = {p.name: sha256_file(p) for p in paths}
        self.scenario_hashes = self.spec.files.hashes()
        models = []
        self.source_episodes = []
        scenario_bound = []
        self.observation_dim = 0
        self.action_count = 0
        for index, path in enumerate(paths):
            payload = _torch_load(path, self.device)
            required = {"source_index", "observation_dim", "action_count", "hidden",
                        "weights", "state_dict"}
            missing = required - set(payload)
            if missing:
                raise ValueError(f"source {path.name} is missing {sorted(missing)}")
            if int(payload["source_index"]) != index:
                raise ValueError(f"source index/order mismatch in {path.name}")
            anchor = np.asarray(payload["weights"], dtype=np.float32)
            if anchor.shape != (5,) or not np.array_equal(anchor, self.anchors[index]):
                raise ValueError(f"source anchor differs from task protocol in {path.name}")
            obs_dim = int(payload["observation_dim"])
            action_count = int(payload["action_count"])
            hidden = int(payload["hidden"])
            if min(obs_dim, action_count, hidden) <= 0:
                raise ValueError(f"invalid source dimensions in {path.name}")
            if index == 0:
                self.observation_dim, self.action_count = obs_dim, action_count
            elif (obs_dim, action_count) != (self.observation_dim, self.action_count):
                raise ValueError("source checkpoint observation/action dimensions differ")
            binding = payload.get("scenario_hashes")
            if binding is not None and binding != self.scenario_hashes:
                raise ValueError(f"source scenario hashes mismatch in {path.name}")
            settings = payload.get("scenario_settings")
            if settings is not None and settings != self.scenario_settings:
                raise ValueError(f"source scenario settings mismatch in {path.name}")
            if payload.get("node_ids") is not None and tuple(payload["node_ids"]) != self.ids:
                raise ValueError(f"source node identities/order mismatch in {path.name}")
            if payload.get("feature_names") is not None and tuple(payload["feature_names"]) != tuple(FEATURE_NAMES):
                raise ValueError(f"source feature definitions mismatch in {path.name}")
            scenario_bound.append(binding is not None and settings is not None)
            self.source_episodes.append(int(payload.get("episodes", 0)))
            models.append(SourceCheckpoint(path, index, obs_dim, action_count, hidden).load(self.device))
        self.source_scenario_binding_verified = all(scenario_bound)
        self.library = FrozenSourcePolicyLibrary(
            models, self.K, self.N, self.action_count, self.horizon, self.device)
        # Keep this alias for collectors/auditors which inspect frozen models.
        self.models = self.library.models
        from .topology import signal_adjacency
        self.adjacency = np.asarray(
            signal_adjacency(self.spec.files.network, self.ids), dtype=bool)
        if self.adjacency.shape != (self.N, self.N):
            raise ValueError("traffic adjacency does not match the declared signal order")

        self.sf_checkpoint = (Path(sf_checkpoint).expanduser().resolve()
                              if sf_checkpoint is not None else None)
        self.sf = None
        self.sf_payload = None
        self.sf_hash = None
        if self.sf_checkpoint is not None:
            from .task_sf import load_task_sf
            self.sf, self.sf_payload = load_task_sf(self.sf_checkpoint, self.device)
            cfg = self.sf.config
            if (int(cfg.state_dim), int(cfg.source_count), int(cfg.task_dim),
                    int(cfg.feature_dim)) != (self.observation_dim, self.K, 5, 5):
                raise ValueError("task SF dimensions differ from the frozen source runtime")
            expected_metadata = {
                "source_hashes": self.source_hashes,
                "protocol_sha256": self.protocol.fingerprint,
                "scenario_hashes": self.scenario_hashes,
                "manifest_sha256": self.manifest_hash,
                "feature_names": list(FEATURE_NAMES),
            }
            for key, expected in expected_metadata.items():
                if self.sf_payload.get(key) != expected:
                    raise ValueError(f"task SF {key} does not match this runtime")
            if not np.array_equal(np.asarray(self.sf_payload.get("adjacency"), dtype=bool),
                                  self.adjacency):
                raise ValueError("task SF graph differs from the current network")
            self.sf.eval()
            self.sf.requires_grad_(False)
            self.sf_hash = sha256_file(self.sf_checkpoint)

        self.weights = np.zeros((self.N, 5), dtype=np.float32)
        self._id_to_index = {node: index for index, node in enumerate(self.ids)}
        self.extractor = TrafficFeatureExtractor()
        self.env = None
        self.state = None
        self.step_index = 0
        self.traffic_seed = None
        self.membership_checks = 0
        self.membership_failures = 0
        self.total_steps = 0
        self._terminated = True
        self._frozen = self._fingerprints()
        if self._frozen["source_files"] != self.source_hashes:
            raise RuntimeError("source checkpoint bytes changed while models were loading")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()
        return False

    def _fingerprints(self) -> dict:
        return {
            "source_files": {p.name: sha256_file(p) for p in self.source_paths},
            "source_parameters": {str(i): _model_hash(model)
                                  for i, model in enumerate(self.models)},
            "sf_file": sha256_file(self.sf_checkpoint) if self.sf_checkpoint else None,
            "sf_parameters": _model_hash(self.sf) if self.sf is not None else None,
            "adjacency": hashlib.sha256(np.ascontiguousarray(self.adjacency).tobytes()).hexdigest(),
            "protocol_file": sha256_file(self.protocol.path),
            "manifest_file": sha256_file(self.manifest.path),
            "scenario_files": self.spec.files.hashes(),
        }

    def frozen_fingerprints(self) -> dict:
        # Return detached nested dictionaries so callers cannot edit the baseline.
        return {key: value.copy() if isinstance(value, dict) else value
                for key, value in self._frozen.items()}

    def audit_frozen(self) -> dict:
        if self._fingerprints() != self._frozen:
            raise RuntimeError("frozen source/SF checkpoint or model parameters changed")
        return {
            "frozen_models_unchanged": True,
            "source_hashes": self.source_hashes.copy(),
            "sf_checkpoint_sha256": self.sf_hash,
            "source_scenario_binding_verified": self.source_scenario_binding_verified,
            "membership_checks": self.membership_checks,
            "membership_failures": self.membership_failures,
            "optimizer_updates": 0,
        }

    def _reward(self, signal) -> float:
        index = self._id_to_index.get(str(signal.id))
        if index is None:
            raise ValueError("reward called for a signal outside the declared node order")
        return self.extractor.reward(signal, weights=self.weights[index])

    def reset(self, task, traffic_seed: int,
              tripinfo_path: str | Path | None = None) -> np.ndarray:
        self.close()
        value = task.get("weights") if isinstance(task, Mapping) else task
        self.weights = validate_task_weights(value, self.N).copy()
        if isinstance(traffic_seed, bool) or int(traffic_seed) != traffic_seed or int(traffic_seed) < 0:
            raise ValueError("traffic seed must be a nonnegative integer")
        self.traffic_seed = int(traffic_seed)
        self.step_index = 0
        self.extractor.reset()
        if tripinfo_path is not None:
            Path(tripinfo_path).expanduser().resolve().parent.mkdir(parents=True, exist_ok=True)
        try:
            self.env = self.spec.build_parallel_env(
                self._reward, tripinfo_path=tripinfo_path,
                sumo_seed=self.traffic_seed)
            ids = tuple(sorted(str(x) for x in getattr(
                self.env, "possible_agents", getattr(self.env, "agents", ()))))
            if ids and ids != self.ids:
                raise ValueError("SUMO controlled signals differ from the declared XML node order")
            result = self.env.reset(seed=self.traffic_seed)
            obs = result[0] if isinstance(result, tuple) and len(result) == 2 else result
            self.state = self._state_array(obs)
            for node in self.ids:
                spaces = getattr(self.env, "action_spaces", None)
                space = spaces[node] if spaces is not None else self.env.action_space(node)
                if int(space.n) != self.action_count:
                    raise ValueError("SUMO action space differs from the frozen source checkpoint")
            self._terminated = False
            return self.state.copy()
        except BaseException:
            self.close()
            raise

    def _state_array(self, obs) -> np.ndarray:
        result = observation_array(obs, self.ids)
        if result.shape != (self.N, self.observation_dim):
            raise ValueError("SUMO observation shape differs from the frozen source checkpoint")
        return result

    def candidate_actions(self, state, step: int) -> np.ndarray:
        if isinstance(step, bool) or int(step) != step or not 0 <= int(step) < self.horizon:
            raise ValueError("source action time must lie inside the episode")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.N, self.observation_dim) or not np.isfinite(state).all():
            raise ValueError("source candidate observations have invalid shape or values")
        state_t = torch.as_tensor(state[None], dtype=torch.float32, device=self.device)
        fraction = torch.tensor([int(step) / self.horizon], dtype=torch.float32, device=self.device)
        with torch.inference_mode():
            result = np.stack([
                model(state_t, fraction)[0].argmax(-1).cpu().numpy()
                for model in self.models], axis=-1).astype(np.int64)
        if result.shape != (self.N, self.K) or np.any(result < 0) or np.any(result >= self.action_count):
            raise RuntimeError("frozen source emitted an invalid candidate action")
        return result

    def sf_values(self, state, step: int) -> np.ndarray:
        if self.sf is None:
            raise RuntimeError("this runtime was not given a task-conditioned SF checkpoint")
        if isinstance(step, bool) or int(step) != step or not 0 <= int(step) <= self.horizon:
            raise ValueError("SF query time must lie inside the finite horizon")
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (self.N, self.observation_dim) or not np.isfinite(state).all():
            raise ValueError("SF observations have invalid shape or values")
        if int(step) == self.horizon:
            return np.zeros((self.N, self.K), dtype=np.float32)
        with torch.inference_mode():
            values = self.sf.candidate_values(
                torch.as_tensor(state[None], dtype=torch.float32, device=self.device),
                torch.as_tensor(self.weights[None], dtype=torch.float32, device=self.device),
                torch.tensor([int(step) / self.horizon], dtype=torch.float32, device=self.device),
                adjacency=self.adjacency)
        result = values[0].detach().cpu().numpy().astype(np.float32)
        if result.shape != (self.N, self.K) or not np.isfinite(result).all():
            raise RuntimeError("task SF emitted invalid candidate values")
        return result

    def step(self, entries: Sequence[int]) -> StepResult:
        if self.env is None or self.state is None or self._terminated:
            raise RuntimeError("reset is required before stepping, and after episode termination")
        indices = np.asarray(entries)
        if (indices.shape != (self.N,) or not np.issubdtype(indices.dtype, np.integer)
                or np.any(indices < 0) or np.any(indices >= self.K)):
            self.membership_failures += 1
            raise ValueError("selector must supply one legal integer source index per node")
        candidates = self.candidate_actions(self.state, self.step_index)
        actions = candidates[np.arange(self.N), indices].astype(np.int64)
        self.membership_checks += self.N
        # Clear only feature values, not pressure/fairness history. Every cache
        # entry below must have been produced by the actual step's reward call.
        self.extractor.feature_cache.clear()
        # If the environment or any integrity check fails after execution, its
        # transition cannot safely be retried from our previous cached state.
        self._terminated = True
        raw = self.env.step({node: int(action) for node, action in zip(self.ids, actions)})
        if len(raw) == 5:
            obs, reward, term, trunc, info = raw
            if isinstance(term, Mapping) and isinstance(trunc, Mapping):
                done = bool(term.get("__all__", False) or trunc.get("__all__", False)) or all(
                    bool(term.get(node, False) or trunc.get(node, False)) for node in self.ids)
            else:
                done = bool(term) or bool(trunc)
        elif len(raw) == 4:
            obs, reward, flag, info = raw
            done = (bool(flag.get("__all__", False)) or all(bool(flag.get(node, False)) for node in self.ids)
                    if isinstance(flag, Mapping) else bool(flag))
        else:
            raise RuntimeError("unsupported parallel environment step signature")
        self.step_index += 1
        self.total_steps += 1
        done = bool(done or self.step_index >= self.horizon)
        if not isinstance(reward, Mapping) or any(node not in reward for node in self.ids):
            raise RuntimeError("environment reward is missing a controlled signal")
        rewards = np.asarray([reward[node] for node in self.ids], dtype=np.float32)
        if rewards.shape != (self.N,) or not np.isfinite(rewards).all():
            raise RuntimeError("environment returned non-finite or malformed rewards")
        if any(node not in self.extractor.feature_cache for node in self.ids):
            raise RuntimeError("actual-step feature cache is missing a controlled signal")
        features = np.stack([self.extractor.feature_cache[node] for node in self.ids]).astype(np.float32)
        if features.shape != (self.N, 5) or not np.isfinite(features).all():
            raise RuntimeError("actual-step feature cache is malformed")
        decomposed = np.sum(features * self.weights, axis=-1)
        if not np.allclose(rewards, decomposed, atol=2e-6, rtol=2e-6):
            raise RuntimeError("actual rewards differ from the cached feature/task dot product")
        missing_obs = not isinstance(obs, Mapping) or any(node not in obs for node in self.ids)
        if missing_obs:
            if not done:
                raise RuntimeError("nonterminal observation is missing a controlled signal")
            next_state = self.state.copy()
        else:
            next_state = self._state_array(obs)
        self.state = next_state
        self._terminated = done
        return StepResult(next_state.copy(), rewards, done, info, features, actions)

    def close(self) -> None:
        env, self.env = self.env, None
        self._terminated = True
        if env is not None:
            env.close()
