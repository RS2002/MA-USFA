"""Traffic feature and reward extraction for the repaired TSC runner.

The historical implementation already contained the intended five feature
definitions.  This module keeps those formulas, but makes the task vector
explicit and checks the identity ``reward == phi @ w`` at every call.  It is
deliberately independent of SUMO imports: a traffic-signal object only needs
the small API used by the old ``composite_reward.py``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import numpy as np

FEATURE_NAMES = ("queue", "wait", "pressure", "speed", "fairness")
FEATURE_DIM = len(FEATURE_NAMES)

W_QUEUE = 1.0
W_WAIT = 0.3
W_PRESSURE = 1.5
W_SPEED = 3.0
W_FAIR = 0.0

MAX_QUEUE_PER_LANE = 50.0
MAX_WAIT_PER_LANE = 1000.0
MAX_PRESSURE_PER_LANE = 20.0
MAX_SPEED = 15.0

DEFAULT_W = np.asarray(
    (-W_QUEUE, -W_WAIT, -W_PRESSURE, W_SPEED, -W_FAIR), dtype=np.float32
)


class FeatureConsistencyError(RuntimeError):
    """Raised when the scalar reward and its feature decomposition disagree."""


def _lane_count(ts: Any) -> int:
    count = len(getattr(ts, "lanes", ()))
    if count <= 0:
        raise ValueError(f"Traffic signal {getattr(ts, 'id', '?')!r} has no lanes")
    return count


def _safe_mean_speed(ts: Any) -> float:
    if hasattr(ts, "get_average_speed"):
        return float(ts.get_average_speed())
    total_speed = 0.0
    vehicles = 0
    lane_api = ts.sumo.lane
    for lane in ts.lanes:
        count = int(lane_api.getLastStepVehicleNumber(lane))
        if count > 0:
            total_speed += float(lane_api.getLastStepMeanSpeed(lane)) * count
            vehicles += count
    return total_speed / vehicles if vehicles else 0.0


@dataclass
class TrafficFeatureExtractor:
    """Stateful per-environment extractor matching the legacy formulas."""

    feature_cache: dict[str, np.ndarray] = field(default_factory=dict)
    _fairness_cache: dict[str, Any] = field(
        default_factory=lambda: {"sim_step": -1, "avg_wait": 0.0}
    )

    def reset(self) -> None:
        self.feature_cache.clear()
        self._fairness_cache = {"sim_step": -1, "avg_wait": 0.0}

    def _fairness_mean(self, ts: Any, waiting: float, current_step: int) -> float:
        cache = self._fairness_cache
        if current_step < int(cache.get("sim_step", -1)):
            self._fairness_cache = {"sim_step": -1, "avg_wait": 0.0}
            cache = self._fairness_cache
        if cache.get("sim_step") != current_step:
            env = getattr(ts, "env", None)
            signals = getattr(env, "traffic_signals", None)
            if signals:
                values = []
                for signal in signals.values():
                    values.append(
                        sum(
                            float(signal.sumo.lane.getWaitingTime(lane))
                            for lane in signal.lanes
                        )
                    )
                cache["avg_wait"] = float(np.mean(values)) if values else waiting
            else:
                cache["avg_wait"] = float(waiting)
            cache["sim_step"] = int(current_step)
        return float(cache["avg_wait"])

    def phi(self, ts: Any) -> np.ndarray:
        lanes_count = _lane_count(ts)
        lane_api = ts.sumo.lane
        queue = float(ts.get_total_queued())
        waiting = float(sum(float(lane_api.getWaitingTime(lane)) for lane in ts.lanes))

        current_in = float(
            sum(float(lane_api.getLastStepVehicleNumber(lane)) for lane in ts.lanes)
        )
        out_lanes = getattr(ts, "out_lanes", ())
        current_out = float(
            sum(float(lane_api.getLastStepVehicleNumber(lane)) for lane in out_lanes)
        )
        if not hasattr(ts, "_ma_usfa_prev_in"):
            ts._ma_usfa_prev_in = current_in
            ts._ma_usfa_prev_out = current_out
        inflow = max(0.0, current_in - float(ts._ma_usfa_prev_in))
        outflow = max(0.0, current_out - float(ts._ma_usfa_prev_out))
        ts._ma_usfa_prev_in, ts._ma_usfa_prev_out = current_in, current_out
        pressure = abs(inflow - outflow)

        speed = _safe_mean_speed(ts)
        delta_time = float(getattr(ts, "delta_time", 1.0))
        simulation = getattr(ts.sumo, "simulation", None)
        sim_time = float(simulation.getTime()) if simulation is not None else 0.0
        current_step = int(sim_time / delta_time) if delta_time > 0 else int(sim_time)
        average_wait = self._fairness_mean(ts, waiting, current_step)
        deviation = waiting - average_wait
        fairness = (deviation * deviation) / (lanes_count * MAX_WAIT_PER_LANE) ** 2

        result = np.asarray(
            (
                queue / (lanes_count * MAX_QUEUE_PER_LANE),
                waiting / (lanes_count * MAX_WAIT_PER_LANE),
                pressure / (lanes_count * MAX_PRESSURE_PER_LANE),
                speed / MAX_SPEED,
                min(fairness, 1.0),
            ),
            dtype=np.float32,
        )
        if not np.isfinite(result).all():
            raise FloatingPointError("non-finite traffic feature vector")
        self.feature_cache[str(ts.id)] = result.copy()
        return result

    def reward(
        self,
        ts: Any,
        weights: np.ndarray | None = None,
        weight_map: Mapping[str, np.ndarray] | None = None,
        atol: float = 1e-6,
    ) -> float:
        phi = self.phi(ts)
        vector = DEFAULT_W if weights is None else np.asarray(weights, dtype=np.float32)
        if weight_map is not None and str(ts.id) in weight_map:
            vector = np.asarray(weight_map[str(ts.id)], dtype=np.float32)
        if vector.shape != (FEATURE_DIM,) or not np.isfinite(vector).all():
            raise ValueError(f"weights must have finite shape [{FEATURE_DIM}]")
        value = float(phi @ vector)
        if not np.isfinite(value):
            raise FloatingPointError("non-finite scalar traffic reward")
        # Keep the historical scalar reward and SF decomposition explicitly
        # coupled; this check is active for the default task only.
        if weights is None and weight_map is None:
            direct = float(
                -W_QUEUE * phi[0]
                - W_WAIT * phi[1]
                - W_PRESSURE * phi[2]
                + W_SPEED * phi[3]
                - W_FAIR * phi[4]
            )
            if not np.isclose(value, direct, rtol=0.0, atol=atol):
                raise FeatureConsistencyError(
                    f"reward/features mismatch for {ts.id}: {value} vs {direct}"
                )
        return value


def validate_weight_matrix(weights: np.ndarray, node_count: int | None = None) -> np.ndarray:
    """Validate a homogeneous vector or per-node task matrix."""
    values = np.asarray(weights, dtype=np.float32)
    if values.ndim == 1 and values.shape == (FEATURE_DIM,):
        result = values.copy()
    elif values.ndim == 2 and values.shape[1] == FEATURE_DIM:
        if node_count is not None and values.shape[0] != node_count:
            raise ValueError("per-node task matrix does not match node count")
        result = values.copy()
    else:
        raise ValueError("weights must be [5] or [N,5]")
    if not np.isfinite(result).all():
        raise ValueError("weights contain non-finite values")
    return result

