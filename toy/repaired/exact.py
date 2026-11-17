"""Float64 finite-horizon evaluation and fixed composition controls.

SF is conditional on a real source continuation, never on the deployed
composer's future actions. All tie breaks use first index in declared order.
"""
from __future__ import annotations
import numpy as np


def canonical_sources(world, sources):
    p = np.asarray(sources, dtype=np.int64)
    if p.ndim == 3 and p.shape[1:] == (world.S, 2):
        p = np.broadcast_to(p[:, None], (len(p), world.horizon, world.S, 2))
    if p.ndim != 4 or p.shape[1:] != (world.horizon, world.S, 2):
        raise ValueError('sources require [K,H,S,2] or [K,S,2].')
    if np.any((p < 0) | (p >= 5)):
        raise ValueError('Primitive action outside 0..4.')
    return p


def policy_indices(world, policy):
    p = np.asarray(policy, dtype=np.int64)
    if p.shape == (world.S, 2):
        p = np.broadcast_to(p, (world.horizon, world.S, 2))
    if p.shape == (world.horizon, world.S, 2):
        p = p[..., 0]*5+p[..., 1]
    if p.shape == (world.S,):
        p = np.broadcast_to(p, (world.horizon, world.S))
    if p.shape != (world.horizon, world.S) or np.any((p < 0) | (p >= world.A)):
        raise ValueError('policy requires [H,S,2], [H,S], [S,2], or [S].')
    return p


def evaluate_policy(world, policy, w):
    acts = policy_indices(world, policy)
    reward = world.reward(world.expected_features, w)
    values = np.zeros((world.horizon+1, world.S))
    ix = np.arange(world.S)
    for t in range(world.horizon-1, -1, -1):
        q = reward + world.gamma*world.expected_next(values[t+1])
        values[t] = q[ix, acts[t]]
    return {'initial_return': float(world.initial @ values[0]), 'values': values}


def evaluate_sources_sf(world, sources):
    p = canonical_sources(world, sources)
    sf = np.zeros((len(p), world.horizon+1, world.S, 2, 5))
    ix = np.arange(world.S)
    for k in range(len(p)):
        for t in range(world.horizon-1, -1, -1):
            acts = p[k, t, :, 0]*5+p[k, t, :, 1]
            sf[k, t] = (world.expected_features + world.gamma*world.expected_next(sf[k, t+1]))[ix, acts]
    return sf


def legacy_sources(world, assignments=((0, 3), (3, 0), (1, 2), (2, 1))):
    """Original library.py infinite-horizon kappa=0 source-action definition.

    Uses its VI tolerance/1000 limit, inverse-based action SF, and *source SF
    readout argmax*, rather than silently substituting a finite-horizon source.
    """
    source_w = np.full((len(assignments), 2, 4), .15)
    psi = np.zeros((len(assignments), 2, 25, 5, 4))
    local_policies = np.zeros((len(assignments), 2, 25), dtype=int)
    phi_next = world.harvest[world.own_next]
    nxt = world.own_next
    for k, assignment in enumerate(assignments):
        for i, goal in enumerate(assignment):
            source_w[k, i, goal] = 1.3
            wi = source_w[k, i]
            v = np.zeros(25)
            for _ in range(1000):
                q = phi_next @ wi + world.gamma*v[nxt]
                vn = q.max(1)
                if np.max(np.abs(vn-v)) < 1e-10:
                    v = vn
                    break
                v = vn
            greedy = (phi_next @ wi + world.gamma*v[nxt]).argmax(1)
            sp = nxt.reshape(-1)
            p = np.zeros((125, 125))
            p[np.arange(125), sp*5+greedy[sp]] = 1.
            inv = np.linalg.inv(np.eye(125)-world.gamma*p)
            for d in range(4):
                psi[k, i, :, :, d] = (inv @ phi_next[..., d].reshape(-1)).reshape(25, 5)
            local_policies[k, i] = (psi[k, i] * wi).sum(-1).argmax(-1)
    sources = np.stack([local_policies[k, np.arange(2)[None], world.state_cells]
                        for k in range(len(assignments))])
    return {'policies': sources, 'local_policies': local_policies,
            'source_weights': source_w, 'legacy_psi': psi,
            'assignments': [list(x) for x in assignments]}


def compose_fixed(world, sources, sf, w, full_features=True):
    """Policies using SF from *world* (which may be stale for deployment).

    Independent uses each agent's snapshot Q: one own-action deviation while
    teammates take source k, then all continue source k. Joint-GPI uses the
    actual joint action Q, maximizing max_k sum_i Q_i^k(s,a_joint).
    """
    p = canonical_sources(world, sources)
    theta = world.theta(w).copy()
    if not full_features:
        theta[:, -1] = 0.
    policies = {name: np.zeros((world.horizon, world.S), dtype=int)
                for name in ('sync', 'primitive_indep', 'true_joint_gpi')}
    ix = np.arange(world.S)
    for t in range(world.horizon):
        joint_q, source_v, snapshots = [], [], []
        for k in range(len(p)):
            qf = world.expected_features + world.gamma*world.expected_next(sf[k, t+1])
            qi = np.sum(qf*theta, -1)  # [S,25,N]
            team = qi.sum(-1)
            joint_q.append(team)
            sa = p[k, t, :, 0]*5+p[k, t, :, 1]
            source_v.append(team[ix, sa])
            own = np.empty((world.S, 2, 5))
            for a in range(5):
                own[:, 0, a] = qi[ix, a*5+p[k, t, :, 1], 0]
                own[:, 1, a] = qi[ix, p[k, t, :, 0]*5+a, 1]
            snapshots.append(own)
        source_v = np.stack(source_v)
        kbest = source_v.argmax(0)
        selected = p[kbest, t, ix]
        policies['sync'][t] = selected[:, 0]*5+selected[:, 1]
        own_actions = np.stack(snapshots).max(0).argmax(-1)
        policies['primitive_indep'][t] = own_actions[:, 0]*5+own_actions[:, 1]
        policies['true_joint_gpi'][t] = np.stack(joint_q).max(0).argmax(-1)
    return policies


def optimal_policy(world, w, sources=None):
    """Model-based finite-horizon optimum, optionally candidate constrained."""
    reward = world.reward(world.expected_features, w)
    v = np.zeros((world.horizon+1, world.S))
    policy = np.zeros((world.horizon, world.S), dtype=int)
    p = None if sources is None else canonical_sources(world, sources)
    ix = np.arange(world.S)
    for t in range(world.horizon-1, -1, -1):
        q = reward+world.gamma*world.expected_next(v[t+1])
        if p is None:
            policy[t] = q.argmax(-1)
        else:
            # Cross-source profiles, not only coherent same-k actions.
            candidates = np.stack([p[k, t, :, 0]*5+p[l, t, :, 1]
                                   for k in range(len(p)) for l in range(len(p))], -1)
            best = q[ix[:, None], candidates].argmax(-1)
            policy[t] = candidates[ix, best]
        v[t] = q[ix, policy[t]]
    return {'initial_return': float(world.initial @ v[0]), 'values': v, 'policy': policy}
