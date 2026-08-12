"""Successor-feature LIBRARY and the composition operators L1 / L2 / joint-GPI.

Backbone for the toy's headline demonstration: in ONE controlled world, show
when L1 > L2, when L2 > L1, and that a learned composer (L2', composers.py) is
>= both -- against a per-task retrain baseline (train_q) and the prior SF->MARL
transfer method (joint-action GPI over the library, papers [2][3]).

A "library" is a set of SOURCE tasks {w^(k)}. For each we compute the EXACT
per-agent successor features psi^k_i(s,a) in R^K on the FACTORIZED kernel (own
moves only) -- the harvest-only SF of the per-agent-optimal SOURCE policy. We
store BOTH psi^k and the source weights w^k, because the three operators differ
precisely in whether they replay the source's ACTIONS or re-optimize actions
for the test task:

  * L1 (synchronized anchor).  Per state, pick a SINGLE shared source k* and
        have every agent replay that source's own greedy action
        a^k_i(s_i) = argmax_a psi^k_i(s_i,a) . w^k_i     (SOURCE weights).
        k* = argmax_k sum_i psi^k_i(s_i, a^k_i) . w_i    (best under TEST task).
        Because a^k is a coherent joint action from one source policy,
        coordination is inherited -- L1 never self-collides (Prop-1 safe).

  * L2 (per-agent GPI).  Each agent independently re-optimizes for the TEST
        task and may mix sources:
        a_i = argmax_a max_k psi^k_i(s_i,a) . w_i         (TEST weights).
        Maximally flexible, but independent argmaxes can COLLIDE (both grab the
        same goal) -> breaks coordination in the paid region.

  * joint_gpi (prior [2][3]).  Treat the team as ONE super-agent: per state
        pick a shared source k and the JOINT action tuple maximizing
        sum_i psi^k_i(s_i,a_i) . w_i  (TEST weights, enumerated over A^N).
        Two honest costs made visible here: (i) it enumerates the JOINT action
        space A^N -- the scalability wall the papers flag (infeasible at TSC's
        196 agents); (ii) psi is collision-blind (harvest-only), so within the
        shared source it re-optimizes actions for w WITHOUT seeing collisions
        -> STALE in the paid region. It sits BETWEEN L1 (more coordinated,
        replays source actions) and L2 (per-agent free source choice).

All operators return a policy callable ``(positions, w) -> [N] actions`` for
env.evaluate. Everything is numpy + exact; deterministic given the env.
"""
import itertools
import numpy as np


# --------------------------------------------------------------------- #
# exact per-agent successor features for one source task
# --------------------------------------------------------------------- #
def exact_psi_source(env, w_src, gamma=0.95, vi_iters=1000, tol=1e-10):
    """Exact harvest-only per-agent SF psi^k_i(s,a) in R^K for source task
    w_src [N,K], on the FACTORIZED (own-move) kernel. Returns psi [N,S,A,K]."""
    N, K = w_src.shape
    S, A = env.n_states, env.n_actions
    nxt = np.zeros((S, A), dtype=int)
    phi_next = np.zeros((S, A, K))
    for s in range(S):
        cell = env.state_to_cell(s)
        for a in range(A):
            ncell = env._step_pos(cell, a)
            nxt[s, a] = env.cell_to_state(ncell)
            phi_next[s, a] = env.phi_of_cell(ncell)

    psi = np.zeros((N, S, A, K))
    for i in range(N):
        wi = w_src[i]
        V = np.zeros(S)
        for _ in range(vi_iters):
            Q = (phi_next @ wi) + gamma * V[nxt]
            Vnew = Q.max(1)
            if np.abs(Vnew - V).max() < tol:
                V = Vnew
                break
            V = Vnew
        greedy = ((phi_next @ wi) + gamma * V[nxt]).argmax(1)   # pi^k_i(s)
        SA = S * A
        sp = nxt.reshape(-1)
        ap = greedy[sp]
        col = sp * A + ap
        P = np.zeros((SA, SA))
        P[np.arange(SA), col] = 1.0
        Minv = np.linalg.inv(np.eye(SA) - gamma * P)
        for d in range(K):
            b = phi_next[:, :, d].reshape(-1)
            psi[i, :, :, d] = (Minv @ b).reshape(S, A)
    return psi


def build_library(env, source_tasks, gamma=0.95):
    """source_tasks: list of [N,K] weight matrices. Returns list of dicts
    {'psi': psi^k [N,S,A,K], 'w': w^k [N,K]}."""
    lib = []
    for w in source_tasks:
        w = np.asarray(w, dtype=float)
        lib.append({'psi': exact_psi_source(env, w, gamma=gamma), 'w': w})
    return lib


# --------------------------------------------------------------------- #
# operators
# --------------------------------------------------------------------- #
def _states(env, positions):
    grid = env.grid
    return np.array([p[0] * grid + p[1] for p in positions])


def make_L2_policy(library, env):
    """Per-agent GPI: a_i = argmax_a max_k psi^k_i(s_i,a) . w_i (test weights)."""
    N = env.n_agents

    def policy(positions, w):
        w = np.asarray(w, dtype=float)
        s = _states(env, positions)
        best = None
        for item in library:
            q = (item['psi'][np.arange(N), s, :, :] * w[:, None, :]).sum(-1)  # [N,A]
            best = q if best is None else np.maximum(best, q)
        return best.argmax(-1)
    return policy


def make_L1_policy(library, env):
    """Synchronized anchor: shared source k*, every agent replays its SOURCE
    greedy action a^k_i = argmax_a psi^k_i(s_i,a) . w^k_i; k* chosen to maximize
    the team's TEST-task value of that coordinated joint action."""
    N = env.n_agents

    def policy(positions, w):
        w = np.asarray(w, dtype=float)
        s = _states(env, positions)
        best_val, best_acts = -np.inf, None
        for item in library:
            psi, wsrc = item['psi'], item['w']
            qsrc = (psi[np.arange(N), s, :, :] * wsrc[:, None, :]).sum(-1)   # [N,A]
            acts = qsrc.argmax(-1)                                           # source-greedy
            qtest = (psi[np.arange(N), s, :, :] * w[:, None, :]).sum(-1)     # [N,A]
            val = qtest[np.arange(N), acts].sum()                           # test value of source action
            if val > best_val:
                best_val, best_acts = val, acts
        return best_acts
    return policy


def make_joint_gpi_policy(library, env):
    """Prior SF->MARL transfer [2][3]: shared source k + JOINT action tuple
    maximizing sum_i psi^k_i(s_i,a_i) . w_i (test weights, enumerated over A^N,
    collision-blind)."""
    A, N = env.n_actions, env.n_agents
    joint_actions = list(itertools.product(range(A), repeat=N))

    def policy(positions, w):
        w = np.asarray(w, dtype=float)
        s = _states(env, positions)
        best_val, best_a = -np.inf, None
        for item in library:
            q = (item['psi'][np.arange(N), s, :, :] * w[:, None, :]).sum(-1)  # [N,A]
            for at in joint_actions:
                val = sum(q[i, at[i]] for i in range(N))
                if val > best_val:
                    best_val, best_a = val, np.array(at, dtype=int)
        return best_a
    return policy
