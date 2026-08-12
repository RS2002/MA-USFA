"""Hand-crafted COMPOSITE reward for the main comparison experiment.

This is the multi-term reward the project used before switching to sumo_rl's
built-in single-metric rewards for the k-hop *analysis*. It is kept for the
MAIN comparison (0-hop-adv CMAT vs baselines) because it couples several
congestion signals into one objective, which is what a deployed controller
would actually optimize.

    r = - w_queue * queue_norm
        - w_wait  * wait_norm
        - w_press * pressure_norm
        + w_speed * speed_norm
        - w_fair  * fairness_norm

All terms are normalized to ~[0,1] per intersection (see the per-lane maxima
below) so the weights below are the true relative importances. Weights and
normalization constants are carried over verbatim from signal/old/v4 so results
stay comparable to the earlier composite-reward runs.

Usage
-----
    from composite_reward import make_composite_reward
    reward_fn = make_composite_reward()      # fresh, self-contained cache
    env = sumo_rl.parallel_env(..., reward_fn=reward_fn)

`make_composite_reward` returns a *callable* with its own closed-over component
and fairness caches, so building several environments never cross-contaminates
state (unlike the old module-global version). The returned callable also carries
a ``.component_cache`` attribute for callers that want the raw per-intersection
term values, but the current agent.py collects its physical metrics from the
env's ``system_*`` info dict and does NOT need it.
"""
import numpy as np

# ---- term weights (verbatim from signal/old/v4) ----
W_QUEUE = 1.0
W_WAIT = 0.3
W_PRESSURE = 1.5
W_SPEED = 3.0
W_FAIR = 0.0

# ---- per-lane normalization maxima (verbatim from signal/old/v4) ----
MAX_QUEUE_PER_LANE = 50       # max queued vehicles / lane
MAX_WAIT_PER_LANE = 1000      # max accumulated waiting time / lane (s)
MAX_PRESSURE_PER_LANE = 20    # max flow / lane / step (veh)
MAX_SPEED = 15.0             # free-flow speed (m/s)


def make_composite_reward():
    """Return a fresh composite reward_fn(ts) with its own cache state."""
    component_cache = {}
    fairness_cache = {'sim_step': -1, 'avg_wait': 0.0}

    def reward_fn(ts):
        lanes_count = len(ts.lanes)

        # queue length
        queue = ts.get_total_queued()
        queue_norm = queue / (lanes_count * MAX_QUEUE_PER_LANE)

        # total accumulated waiting time
        waiting = sum(ts.sumo.lane.getWaitingTime(l) for l in ts.lanes)
        wait_norm = waiting / (lanes_count * MAX_WAIT_PER_LANE)

        # pressure via inflow/outflow differencing (per-ts persistent counters)
        current_in = sum(ts.sumo.lane.getLastStepVehicleNumber(l) for l in ts.lanes)
        out_lanes = getattr(ts, 'out_lanes', [])
        current_out = sum(ts.sumo.lane.getLastStepVehicleNumber(l) for l in out_lanes)
        if not hasattr(ts, '_prev_in'):
            ts._prev_in = current_in
            ts._prev_out = current_out
        inflow = max(0, current_in - ts._prev_in)
        outflow = max(0, current_out - ts._prev_out)
        ts._prev_in, ts._prev_out = current_in, current_out
        pressure = abs(inflow - outflow)
        pressure_norm = pressure / (lanes_count * MAX_PRESSURE_PER_LANE)

        # average speed (vehicle-weighted over the intersection's lanes)
        if hasattr(ts, 'get_average_speed'):
            speed = ts.get_average_speed()
        else:
            total_speed = 0.0
            veh_cnt = 0
            for lane in ts.lanes:
                veh = ts.sumo.lane.getLastStepVehicleNumber(lane)
                if veh > 0:
                    total_speed += ts.sumo.lane.getLastStepMeanSpeed(lane) * veh
                    veh_cnt += veh
            speed = total_speed / veh_cnt if veh_cnt > 0 else 0.0
        speed_norm = speed / MAX_SPEED

        # fairness: squared deviation of this ts's waiting from the net-wide mean
        current_step = int(ts.sumo.simulation.getTime() / ts.delta_time)
        if current_step < fairness_cache.get('sim_step', -1):   # new episode -> reset
            fairness_cache.clear()
            fairness_cache['sim_step'] = -1
            fairness_cache['avg_wait'] = 0.0
        if fairness_cache.get('sim_step') != current_step:
            if hasattr(ts, 'env') and hasattr(ts.env, 'traffic_signals'):
                all_ts = list(ts.env.traffic_signals.values())
                if len(all_ts) > 1:
                    waitings = [sum(sig.sumo.lane.getWaitingTime(l) for l in sig.lanes)
                                for sig in all_ts]
                    fairness_cache['avg_wait'] = float(np.mean(waitings))
                else:
                    fairness_cache['avg_wait'] = waiting
            else:
                fairness_cache['avg_wait'] = waiting
            fairness_cache['sim_step'] = current_step

        deviation = waiting - fairness_cache['avg_wait']
        fairness_penalty = deviation ** 2
        fairness_denom = (lanes_count * MAX_WAIT_PER_LANE) ** 2
        fairness_norm = fairness_penalty / fairness_denom if fairness_denom > 0 else 0.0
        fairness_norm = min(fairness_norm, 1.0)

        reward = (
            - W_QUEUE * queue_norm
            - W_WAIT * wait_norm
            - W_PRESSURE * pressure_norm
            + W_SPEED * speed_norm
            - W_FAIR * fairness_norm
        )

        component_cache[ts.id] = {
            'queue': queue, 'waiting': waiting, 'pressure': pressure,
            'speed': speed, 'fairness': fairness_penalty,
        }
        return reward

    reward_fn.component_cache = component_cache
    return reward_fn


# ===================================================================== #
# Successor-feature decomposition of the composite reward
# ===================================================================== #
# The composite reward is ALREADY a linear combination of normalized
# congestion signals:
#     r = phi . w,   phi = [queue, wait, pressure, speed, fairness]_norm
#     w   = [-W_QUEUE, -W_WAIT, -W_PRESSURE, +W_SPEED, -W_FAIR]
# so the five normalized terms are exactly the successor features phi and the
# signed term weights are the task vector w. Exposing phi lets us (a) learn
# w-free successor features psi (USFA) whose value transfers to any weighting by
# the closed form Q = psi . w, and (b) retrain a per-task baseline on r = phi.w
# for any specific w. This is the SF hook the paper's TSC experiment needs; the
# scalar-reward IDDQN path above is untouched (it stays the per-task baseline).

FEATURE_NAMES = ("queue", "wait", "pressure", "speed", "fairness")
FEATURE_DIM = len(FEATURE_NAMES)

# The default task vector: signs + importances that reproduce the scalar
# composite reward EXACTLY (phi . DEFAULT_W == the reward above).
DEFAULT_W = np.array([-W_QUEUE, -W_WAIT, -W_PRESSURE, +W_SPEED, -W_FAIR],
                     dtype=np.float32)


def _phi_of_ts(ts, fairness_cache):
    """Compute the 5-dim normalized successor-feature vector for one signal.

    Identical term math to ``make_composite_reward`` (queue / wait / pressure /
    speed / fairness), returned as a raw feature vector phi in [~0, 1]^5 BEFORE
    any weighting -- the weighting is applied by the caller as phi . w."""
    lanes_count = len(ts.lanes)

    queue = ts.get_total_queued()
    queue_norm = queue / (lanes_count * MAX_QUEUE_PER_LANE)

    waiting = sum(ts.sumo.lane.getWaitingTime(l) for l in ts.lanes)
    wait_norm = waiting / (lanes_count * MAX_WAIT_PER_LANE)

    current_in = sum(ts.sumo.lane.getLastStepVehicleNumber(l) for l in ts.lanes)
    out_lanes = getattr(ts, 'out_lanes', [])
    current_out = sum(ts.sumo.lane.getLastStepVehicleNumber(l) for l in out_lanes)
    if not hasattr(ts, '_prev_in'):
        ts._prev_in = current_in
        ts._prev_out = current_out
    inflow = max(0, current_in - ts._prev_in)
    outflow = max(0, current_out - ts._prev_out)
    ts._prev_in, ts._prev_out = current_in, current_out
    pressure = abs(inflow - outflow)
    pressure_norm = pressure / (lanes_count * MAX_PRESSURE_PER_LANE)

    if hasattr(ts, 'get_average_speed'):
        speed = ts.get_average_speed()
    else:
        total_speed, veh_cnt = 0.0, 0
        for lane in ts.lanes:
            veh = ts.sumo.lane.getLastStepVehicleNumber(lane)
            if veh > 0:
                total_speed += ts.sumo.lane.getLastStepMeanSpeed(lane) * veh
                veh_cnt += veh
        speed = total_speed / veh_cnt if veh_cnt > 0 else 0.0
    speed_norm = speed / MAX_SPEED

    current_step = int(ts.sumo.simulation.getTime() / ts.delta_time)
    if current_step < fairness_cache.get('sim_step', -1):
        fairness_cache.clear()
        fairness_cache['sim_step'] = -1
        fairness_cache['avg_wait'] = 0.0
    if fairness_cache.get('sim_step') != current_step:
        if hasattr(ts, 'env') and hasattr(ts.env, 'traffic_signals'):
            all_ts = list(ts.env.traffic_signals.values())
            if len(all_ts) > 1:
                waitings = [sum(sig.sumo.lane.getWaitingTime(l) for l in sig.lanes)
                            for sig in all_ts]
                fairness_cache['avg_wait'] = float(np.mean(waitings))
            else:
                fairness_cache['avg_wait'] = waiting
        else:
            fairness_cache['avg_wait'] = waiting
        fairness_cache['sim_step'] = current_step
    deviation = waiting - fairness_cache['avg_wait']
    fairness_denom = (lanes_count * MAX_WAIT_PER_LANE) ** 2
    fairness_norm = (deviation ** 2) / fairness_denom if fairness_denom > 0 else 0.0
    fairness_norm = min(fairness_norm, 1.0)

    return np.array([queue_norm, wait_norm, pressure_norm, speed_norm,
                     fairness_norm], dtype=np.float32)


def make_sf_reward(w=None):
    """Return an SF reward_fn(ts) = phi(ts) . w that also caches phi per signal.

    ``w`` is the (signed) task vector over FEATURE_NAMES; ``None`` uses
    ``DEFAULT_W`` (which reproduces the scalar composite reward exactly). The
    returned callable carries a ``.feature_cache`` dict mapping ts.id -> the raw
    phi vector of the latest step, so the training agent can read phi for the
    SF-TD update after each env step. Pass a DIFFERENT ``w`` (or resample it per
    episode via ``.set_w``) to instantiate a new task in the same environment.

    For the w-HETEROGENEOUS scenario, call ``.set_w_map({ts_id: w_row})`` to give
    each intersection its OWN task vector; when a w-map is set the per-ts reward
    is ``phi(ts) . w_map[ts.id]`` (falling back to the shared w for any id not in
    the map). This keeps the environment's reward signal consistent with the
    per-agent w the transfer/composer/L1 operators optimize."""
    w = DEFAULT_W.copy() if w is None else np.asarray(w, dtype=np.float32)
    state = {'w': w, 'w_map': None}
    feature_cache = {}
    fairness_cache = {'sim_step': -1, 'avg_wait': 0.0}

    def reward_fn(ts):
        phi = _phi_of_ts(ts, fairness_cache)
        feature_cache[ts.id] = phi
        wmap = state['w_map']
        w_here = wmap[ts.id] if (wmap is not None and ts.id in wmap) else state['w']
        return float(phi @ w_here)

    def set_w(new_w):
        state['w'][:] = np.asarray(new_w, dtype=np.float32)

    def set_w_map(w_map):
        """Per-intersection weights: dict ts_id -> length-K vector (or None to
        revert to the shared homogeneous w)."""
        state['w_map'] = (None if w_map is None else
                          {k: np.asarray(v, dtype=np.float32) for k, v in w_map.items()})

    reward_fn.feature_cache = feature_cache
    reward_fn.set_w = set_w
    reward_fn.set_w_map = set_w_map
    reward_fn.w = state['w']
    return reward_fn


def sample_task(rng, base=None, sign=None, scale=(0.5, 1.5)):
    """Sample a random task vector for w-free (USFA) SF training.

    Keeps the FIXED signs of DEFAULT_W (queue/wait/pressure/fairness are costs,
    speed is a gain) and only jitters the per-term IMPORTANCE magnitudes within
    ``scale`` x their default, so every sampled task is a physically sensible
    congestion objective. Returns a length-FEATURE_DIM float32 vector."""
    base = np.abs(DEFAULT_W) if base is None else np.abs(base)
    sign = np.sign(DEFAULT_W) if sign is None else sign
    lo, hi = scale
    mag = base * rng.uniform(lo, hi, size=FEATURE_DIM)
    return (sign * mag).astype(np.float32)


def sample_hetero_task(rng, num_agents, scale=(0.5, 1.5)):
    """Random PER-INTERSECTION task matrix W of shape [num_agents, FEATURE_DIM].

    Each intersection draws its OWN importance magnitudes independently (fixed
    DEFAULT_W signs), so the network is asked to optimize a spatially
    heterogeneous objective -- different corners of the city care about
    different congestion terms. This is the w-heterogeneous TSC scenario: the
    closed-form transfer must serve every node's own w simultaneously, and the
    neighbour composer can additionally condition on neighbours' w."""
    return np.stack([sample_task(rng, scale=scale) for _ in range(num_agents)],
                    axis=0).astype(np.float32)


# ===================================================================== #
# Source library for the L1 synchronized-anchor operator
# ===================================================================== #
# L1 needs a small discrete set of SOURCE task profiles to pick a single shared
# anchor from. Each profile keeps the DEFAULT_W signs (physically sensible) but
# emphasizes ONE congestion term (plus a balanced profile), so the anchor
# choice = "which single congestion priority does the whole team commit to this
# step". The per-agent source policy is argmax_a psi_i . w^k in CLOSED FORM from
# the already-trained w-free psi -- L1 needs NO extra training, only the frozen
# psi, so it is fully feasible at 196 agents (O(K*N) per step, no A^N).
def make_source_library(emphasis=3.0, lib_size=None):
    """Return an [K, FEATURE_DIM] matrix of K source task profiles.

    Profiles: one specialist per non-zero DEFAULT_W term (queue / wait /
    pressure / speed) with that term's magnitude boosted by ``emphasis``, plus
    the balanced DEFAULT_W itself. Signs are always DEFAULT_W's.

    ``lib_size`` controls how many sources are returned (for K-sweep experiments).
    The full library has 5 profiles: 1 balanced + 4 specialists (queue/wait/
    pressure/speed). lib_size truncates from the front so that:
      K=1: balanced only
      K=2: balanced + queue-specialist
      K=3: balanced + queue + wait specialists
      K=4: balanced + queue + wait + pressure specialists   (default)
      K=5: all five (adds speed-specialist)
    If lib_size is None (default), all profiles are returned."""
    sign = np.sign(DEFAULT_W)
    base = np.abs(DEFAULT_W)
    sources = [DEFAULT_W.copy()]                        # balanced anchor
    for j in range(FEATURE_DIM):
        if base[j] == 0:                               # skip inert terms (fairness)
            continue
        mag = base.copy()
        mag[j] = base[j] * emphasis                    # specialist on term j
        sources.append((sign * mag).astype(np.float32))
    lib = np.stack(sources, axis=0).astype(np.float32)   # [K_full, FEATURE_DIM]
    if lib_size is not None:
        assert 1 <= lib_size <= len(lib), \
            f"lib_size={lib_size} out of range [1, {len(lib)}]"
        lib = lib[:lib_size]
    return lib
