"""Frozen source-entry library and strict source execution."""
from __future__ import annotations
from dataclasses import dataclass
import hashlib, json
from pathlib import Path
from typing import Any, Sequence
import numpy as np

def _as_action_array(value: object) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 3:
        raise ValueError('source actions must have shape [source,time,node]')
    if not np.issubdtype(arr.dtype, np.integer):
        if not np.all(np.equal(arr, np.floor(arr))):
            raise ValueError('source actions must be integer-valued')
        arr = arr.astype(np.int64)
    else:
        arr = arr.astype(np.int64, copy=False)
    return np.ascontiguousarray(arr)

@dataclass(frozen=True)
class FrozenSourceLibrary:
    """Immutable primitive actions for K frozen source entries."""
    actions: object
    source_count: int
    horizon: int
    node_count: int
    action_count: int
    source_metadata: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        arr = _as_action_array(self.actions).copy()
        arr.setflags(write=False)
        object.__setattr__(self, 'actions', arr)
        expected = (self.source_count, self.horizon, self.node_count)
        if expected != tuple(arr.shape):
            raise ValueError(f'source actions must have shape {expected}, got {arr.shape}')
        if min(self.source_count, self.horizon, self.node_count, self.action_count) <= 0:
            raise ValueError('source-library dimensions must be positive')
        if np.any(arr < 0) or np.any(arr >= self.action_count):
            raise ValueError('source actions contain an invalid primitive action')
        if self.source_metadata and len(self.source_metadata) != self.source_count:
            raise ValueError('source_metadata must contain one record per source')

    @classmethod
    def from_array(cls, actions: object, *, action_count: int,
                   source_metadata: Sequence[dict[str, Any]] | None = None):
        arr = _as_action_array(actions)
        return cls(arr, int(arr.shape[0]), int(arr.shape[1]), int(arr.shape[2]),
                   int(action_count), tuple(dict(item) for item in (source_metadata or ())))

    @classmethod
    def load_npz(cls, path: str | Path) -> 'FrozenSourceLibrary':
        source_path = Path(path).expanduser().resolve()
        with np.load(source_path, allow_pickle=False) as data:
            if 'actions' not in data or 'action_count' not in data:
                raise ValueError('source npz needs actions and action_count arrays')
            actions = data['actions']
            action_count = int(np.asarray(data['action_count']).reshape(()))
            metadata = ()
            if 'metadata_json' in data:
                parsed = json.loads(str(np.asarray(data['metadata_json']).reshape(())))
                if not isinstance(parsed, list) or not all(isinstance(x, dict) for x in parsed):
                    raise ValueError('metadata_json must be a JSON list of objects')
                metadata = tuple(parsed)
        return cls.from_array(actions, action_count=action_count, source_metadata=metadata)

    def save_npz(self, path: str | Path) -> str:
        target = Path(path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(self.source_metadata, sort_keys=True, separators=(',', ':'))
        np.savez_compressed(target, actions=self.actions,
                            action_count=np.asarray(self.action_count, dtype=np.int64),
                            metadata_json=np.asarray(metadata))
        return hashlib.sha256(target.read_bytes()).hexdigest()

    def action(self, source_index: int, time_index: int, node_index: int) -> int:
        source_index, time_index, node_index = map(int, (source_index, time_index, node_index))
        if not 0 <= source_index < self.source_count:
            raise IndexError('source index out of range')
        if not 0 <= time_index < self.horizon:
            raise IndexError('time index out of range')
        if not 0 <= node_index < self.node_count:
            raise IndexError('node index out of range')
        return int(self.actions[source_index, time_index, node_index])

    def actions_for_entries(self, source_indices: Sequence[int], time_index: int) -> np.ndarray:
        indices = validate_source_indices(source_indices, self.source_count)
        if len(indices) != self.node_count:
            raise ValueError('one source index is required for every controlled node')
        time_index = int(time_index)
        if not 0 <= time_index < self.horizon:
            raise IndexError('time index out of range')
        result = np.asarray(self.actions[indices, time_index, np.arange(self.node_count)], dtype=np.int64)
        if np.any(result < 0) or np.any(result >= self.action_count):
            raise ValueError('strict execution produced an invalid primitive action')
        return np.ascontiguousarray(result)

    def membership_check(self, source_indices: Sequence[int], actions: Sequence[int], time_index: int) -> bool:
        expected = self.actions_for_entries(source_indices, time_index)
        actual = np.asarray(actions, dtype=np.int64).reshape(-1)
        return actual.shape == expected.shape and bool(np.array_equal(actual, expected))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(self.actions.tobytes(order='C'))
        digest.update(str(self.action_count).encode())
        digest.update(json.dumps(self.source_metadata, sort_keys=True).encode())
        return digest.hexdigest()

def validate_source_indices(indices: Sequence[int], source_count: int) -> tuple[int, ...]:
    values = tuple(int(value) for value in indices)
    if any(value < 0 or value >= int(source_count) for value in values):
        raise ValueError('selector emitted a source index outside the frozen library')
    return values

def execute_entries(library: FrozenSourceLibrary, source_indices: Sequence[int], time_index: int) -> tuple[int, ...]:
    return tuple(library.actions_for_entries(source_indices, time_index).tolist())

def k1_entry_consistency(library: FrozenSourceLibrary, time_index: int = 0) -> bool:
    if library.source_count != 1:
        raise ValueError('K=1 consistency requires a one-entry library')
    selected = library.actions_for_entries((0,) * library.node_count, time_index)
    return bool(np.array_equal(selected, library.actions[0, int(time_index), :]))


class FrozenSourcePolicyLibrary:
    """Frozen callable source library used by the SUMO collector.

    Unlike the compact ``FrozenSourceLibrary`` tensor used in contract tests,
    this class evaluates a frozen source checkpoint on the current traffic
    observation.  The selector still emits only source indices; no residual or
    primitive-action correction is accepted.
    """

    def __init__(self, models, source_count: int, node_count: int,
                 action_count: int, horizon: int, device: str = "cpu"):
        if len(models) != int(source_count):
            raise ValueError("one frozen model is required per source index")
        self.models = tuple(models)
        self.source_count = int(source_count)
        self.node_count = int(node_count)
        self.action_count = int(action_count)
        self.horizon = int(horizon)
        self.device = str(device)
        for model in self.models:
            model.eval()
            for parameter in model.parameters():
                parameter.requires_grad_(False)

    @classmethod
    def load_from_dir(cls, path: str | Path, node_count: int,
                      action_count: int, horizon: int, device: str = "cpu"):
        from .source_policy import SourceCheckpoint
        root = Path(path).expanduser().resolve()
        checkpoints = sorted(root.glob("source_*.pt"))
        if not checkpoints:
            raise FileNotFoundError(f"no source checkpoints in {root}")
        import torch
        payload = torch.load(checkpoints[0], map_location=device, weights_only=False)
        observation_dim = int(payload["observation_dim"])
        hidden = int(payload["hidden"])
        if int(payload["action_count"]) != int(action_count):
            raise ValueError("source checkpoint action_count differs from manifest")
        models = [SourceCheckpoint(item, index, observation_dim, action_count, hidden).load(device)
                  for index, item in enumerate(checkpoints)]
        return cls(models, len(models), node_count, action_count, horizon, device)

    def actions_for_entries(self, source_indices: Sequence[int], observations,
                            time_index: int) -> np.ndarray:
        import torch
        indices = validate_source_indices(source_indices, self.source_count)
        obs = np.asarray(observations, dtype=np.float32)
        if obs.ndim != 2 or obs.shape[0] != self.node_count:
            raise ValueError("observations must have shape [node,feature]")
        if not 0 <= int(time_index) < self.horizon:
            raise IndexError("time index out of range")
        state = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)
        fraction = torch.full((1,), float(time_index) / max(1, self.horizon), device=self.device)
        actions = np.empty(self.node_count, dtype=np.int64)
        with torch.no_grad():
            for source in range(self.source_count):
                mask = np.asarray(indices) == source
                if np.any(mask):
                    actions[mask] = self.models[source](state, fraction)[0].argmax(-1).cpu().numpy()[mask]
        if np.any(actions < 0) or np.any(actions >= self.action_count):
            raise ValueError("source policy emitted an invalid primitive action")
        return actions

    def membership_check(self, source_indices: Sequence[int], observations,
                         actions, time_index: int) -> bool:
        expected = self.actions_for_entries(source_indices, observations, time_index)
        actual = np.asarray(actions, dtype=np.int64).reshape(-1)
        return actual.shape == expected.shape and bool(np.array_equal(actual, expected))

@dataclass(frozen=True)
class CallableSourceLibrary:
    """Frozen, state-dependent source policies.

    ``FrozenSourceLibrary`` is useful for deterministic contract tests and for
    cached trajectories.  A real traffic source must react to the current
    SUMO observation, so the server runner uses this class instead.  Each
    policy is a frozen ``SourceQNet``-compatible module that maps a complete
    ``[1,N,D]`` observation tensor and a time fraction to ``[1,N,A]`` Q
    values.  The selector still emits only a source index; this class is the
    only component allowed to turn that index into primitive actions.
    """

    policies: tuple[Any, ...]
    node_count: int
    action_count: int
    observation_dim: int
    horizon: int
    device: str = "cpu"
    source_metadata: tuple[dict[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not self.policies:
            raise ValueError("at least one frozen source policy is required")
        if min(self.node_count, self.action_count, self.observation_dim, self.horizon) <= 0:
            raise ValueError("callable source dimensions must be positive")
        if self.source_metadata and len(self.source_metadata) != len(self.policies):
            raise ValueError("source_metadata must contain one record per policy")
        # Freeze the policy objects at the boundary.  The runner must not
        # silently update a source after it becomes part of the library.
        for policy in self.policies:
            if not hasattr(policy, "eval") or not hasattr(policy, "parameters"):
                raise TypeError("source policies must be torch modules")
            policy.eval()
            for parameter in policy.parameters():
                parameter.requires_grad_(False)

    @property
    def source_count(self) -> int:
        return len(self.policies)

    @classmethod
    def from_checkpoints(
        cls,
        checkpoints: Sequence[str | Path],
        *,
        device: str = "cpu",
        node_count: int,
        horizon: int,
    ) -> "CallableSourceLibrary":
        """Load and freeze ``SourceQNet`` checkpoints by source index.

        The checkpoint format is the one emitted by ``train_source.py``.  All
        policies must have identical observation/action dimensions.  Source
        indices are assigned by list order and checked against each checkpoint
        payload's recorded ``source_index``.
        """
        from .source_policy import SourceCheckpoint

        loaded = []
        metadata = []
        for expected_index, raw_path in enumerate(checkpoints):
            path = Path(raw_path).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(path)
            payload = __import__("torch").load(path, map_location=device, weights_only=False)
            if int(payload.get("source_index", -1)) != expected_index:
                raise ValueError(f"checkpoint {path} has wrong source_index")
            model = SourceCheckpoint(
                path=path,
                source_index=expected_index,
                observation_dim=int(payload["observation_dim"]),
                action_count=int(payload["action_count"]),
                hidden=int(payload["hidden"]),
            ).load(device=device)
            loaded.append(model)
            metadata.append({"path": str(path), "sha256": _file_hash(path),
                             "source_index": expected_index})
        if not loaded:
            raise ValueError("no source checkpoints supplied")
        first = loaded[0]
        for model in loaded[1:]:
            if (model.observation_dim != first.observation_dim
                    or model.action_count != first.action_count):
                raise ValueError("source checkpoints disagree on model dimensions")
        return cls(tuple(loaded), int(node_count), int(first.action_count),
                   int(first.observation_dim), int(horizon), str(device), tuple(metadata))

    def actions_for_observation(
        self,
        source_indices: Sequence[int],
        observations: np.ndarray,
        time_index: int,
    ) -> np.ndarray:
        """Return the primitive action from each selected source policy.

        The observation is shared with every source policy, but each node is
        indexed into the policy selected for that node.  No residual or
        primitive-action argmax outside the selected source is permitted.
        """
        import torch

        indices = validate_source_indices(source_indices, self.source_count)
        obs = np.asarray(observations, dtype=np.float32)
        if obs.shape != (self.node_count, self.observation_dim):
            raise ValueError(
                f"observations must have shape {(self.node_count, self.observation_dim)}, got {obs.shape}"
            )
        time_index = int(time_index)
        if not 0 <= time_index < self.horizon:
            raise IndexError("time index out of range")
        obs_t = torch.as_tensor(obs[None], dtype=torch.float32, device=self.device)
        time_t = torch.full((1,), float(time_index) / max(1, self.horizon),
                            dtype=torch.float32, device=self.device)
        per_source: list[np.ndarray] = []
        with torch.no_grad():
            for policy in self.policies:
                q = policy(obs_t, time_t)
                if q.shape != (1, self.node_count, self.action_count):
                    raise ValueError("source policy returned an unexpected Q shape")
                per_source.append(q[0].argmax(-1).detach().cpu().numpy().astype(np.int64))
        stacked = np.stack(per_source, axis=0)
        result = stacked[np.asarray(indices, dtype=np.int64), np.arange(self.node_count)]
        if np.any(result < 0) or np.any(result >= self.action_count):
            raise ValueError("callable source produced an invalid primitive action")
        return np.ascontiguousarray(result)

    def membership_check_observation(
        self,
        source_indices: Sequence[int],
        observations: np.ndarray,
        actions: Sequence[int],
        time_index: int,
    ) -> bool:
        expected = self.actions_for_observation(source_indices, observations, time_index)
        actual = np.asarray(actions, dtype=np.int64).reshape(-1)
        return actual.shape == expected.shape and bool(np.array_equal(actual, expected))

    def fingerprint(self) -> str:
        digest = hashlib.sha256()
        digest.update(str((self.source_count, self.node_count, self.action_count,
                           self.observation_dim, self.horizon)).encode())
        for policy in self.policies:
            state = policy.state_dict()
            for name in sorted(state):
                digest.update(name.encode())
                digest.update(state[name].detach().cpu().contiguous().numpy().tobytes())
        digest.update(json.dumps(self.source_metadata, sort_keys=True).encode())
        return digest.hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
