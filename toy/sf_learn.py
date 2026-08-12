"""Tabular Successor-Feature learning for SFWorld (numpy only).

Implements the two training modes the paper needs:

* ``train_sf(env, w, kappa)`` -- per-task training (the "retrain from
  scratch" baseline every existing method needs for every new task).
* ``train_sf_usfa(env, sampler, kappa)`` -- w-free training: a fresh random
  task vector is drawn every episode, so ``psi`` learns the *feature-based*
  value without committing to any task (Borsa et al. 2019, USFA). One run
  produces a ``psi`` that transfers to any task by the closed-form dot product
  ``Q = psi . w`` -- the free-region mechanism.

Both use the same SF-Q-learning update. The two crucial knobs:

* ``kappa`` -- the transition-coupling level *experienced during training*.
  The USFA features are learned at the anchor regime ``kappa=0`` (the free
  region, where the theory guarantees SF transfer is valid). A per-task
  retrain in the paid region trains at that task's ``kappa`` and therefore
  adapts to the coupled kernel.
* per-task training draws transitions from the *task's* induced joint policy;
  w-free training draws them from a *mixture* over random tasks. In the free
  region the kernel is task-independent, so the mixture is harmless and
  ``psi`` stays valid for every ``w``. In the paid region the kernel depends
  on the task, so the anchor-regime ``psi`` is stale for any specific task --
  the value-validity failure the paper predicts (and T3 measures).

Policies extracted from ``psi``:

* L2: per-agent greedy ``argmax_a psi_i(s_i,a).w_i`` (independent learners).
* L1: shared-index greedy ``argmax_a sum_i psi_i(s_i,a).w_i`` -- all agents
  take the same action index (the unconditionally-safe synchronized anchor).
"""

import numpy as np

DEFAULT_EPISODES = 2000
DEFAULT_ALPHA = 0.05


def _one_episode(env, psi, w, kappa, alpha, gamma, eps, rng):
    """Run one SF-Q-learning episode on the environment."""
    n, S, A, K = psi.shape
    cells = rng.choice(env.n_states, n, replace=False)
    positions = [env.state_to_cell(c) for c in cells]
    for _ in range(env.horizon):
        s = np.array([env.cell_to_state(p) for p in positions])
        q = (psi[range(n), s, :, :] * w[:, None, :]).sum(-1)      # [N, A]
        greedy = q.argmax(-1)
        explore = rng.rand(n) < eps
        a = np.where(explore, rng.randint(0, A, size=n), greedy)

        positions, _ = env.step_intended(positions, a, kappa, rng=rng)
        s2 = np.array([env.cell_to_state(p) for p in positions])
        phi = np.array([env.phi_of_cell(p) for p in positions])    # [N, K]

        q2 = (psi[range(n), s2, :, :] * w[:, None, :]).sum(-1)     # [N, A]
        a2 = q2.argmax(-1)
        for i in range(n):
            delta = phi[i] @ w[i] + gamma * (psi[i, s2[i], a2[i]] @ w[i]) \
                    - (psi[i, s[i], a[i]] @ w[i])
            psi[i, s[i], a[i]] += alpha * delta * w[i]
    return psi


def train_sf(env, w, kappa=0.0, episodes=DEFAULT_EPISODES,
             alpha=DEFAULT_ALPHA, gamma=0.95, eps_start=1.0, eps_end=0.05,
             eps_decay=800, seed=0):
    """Per-task SF training with a fixed task vector ``w`` at coupling
    ``kappa``. Returns psi [N, S, A, K].

    NOTE: this learns the HARVEST value only (psi . w); it is used for the
    w-free USFA features and for the closed-form transfer policy. It cannot
    represent the collision penalty (which lives outside phi), so it is NOT
    the per-task retrain baseline in the paid region -- use ``train_q`` for
    that (a plain per-agent Q-learner on the true, penalty-inclusive reward)."""
    rng = np.random.RandomState(seed)
    n, K = w.shape
    psi = np.zeros((n, env.n_states, env.n_actions, K))
    for ep in range(episodes):
        eps = max(eps_end, eps_start + (eps_end - eps_start) * ep / max(1, eps_decay))
        _one_episode(env, psi, w, kappa, alpha, gamma, eps, rng)
    return psi


def train_q(env, w, kappa=0.0, episodes=DEFAULT_EPISODES, alpha=0.1,
            gamma=0.95, eps_start=1.0, eps_end=0.05, eps_decay=800, seed=0):
    """Per-task per-agent tabular Q-learning on the TRUE reward
    (harvest phi.w MINUS the collision penalty) -- the honest retrain baseline
    in the paid region.

    Each agent learns Q_i(s_i, a_i) on its own penalty-inclusive reward. The
    per-agent table is indexed by own state only, so it cannot condition on
    whether a teammate contests its target cell; it therefore learns the
    AVERAGE collision cost of each state-action (statistical deferral), which
    already beats the collision-blind closed-form transfer. The gap that a
    global-information composer (attn/comm) can additionally recover is the
    CONDITIONAL part -- avoiding a collision only when a teammate actually
    contests. Returns Q [N, S, A]."""
    rng = np.random.RandomState(seed)
    n, K = w.shape
    pen = env.collision_penalty
    Q = np.zeros((n, env.n_states, env.n_actions))
    for ep in range(episodes):
        eps = max(eps_end, eps_start + (eps_end - eps_start) * ep / max(1, eps_decay))
        cells = rng.choice(env.n_states, n, replace=False)
        positions = [env.state_to_cell(c) for c in cells]
        for _ in range(env.horizon):
            s = np.array([env.cell_to_state(p) for p in positions])
            greedy = Q[range(n), s, :].argmax(-1)
            explore = rng.rand(n) < eps
            a = np.where(explore, rng.randint(0, env.n_actions, size=n), greedy)
            positions, blocked = env.step_intended(positions, a, kappa, rng=rng)
            s2 = np.array([env.cell_to_state(p) for p in positions])
            r = np.array([float(env.phi_of_cell(positions[i]) @ w[i]) - pen * blocked[i]
                          for i in range(n)])
            for i in range(n):
                td = r[i] + gamma * Q[i, s2[i]].max() - Q[i, s[i], a[i]]
                Q[i, s[i], a[i]] += alpha * td
    return Q


def make_q_policy(Q):
    """Per-agent greedy on a plain Q-table: a_i = argmax_a Q_i(s_i, a)."""
    n, S, A = Q.shape
    grid = int(round(S ** 0.5))

    def policy(positions, w):
        s = np.array([p[0] * grid + p[1] for p in positions])
        return Q[range(n), s, :].argmax(-1)
    return policy


def train_sf_usfa(env, sampler, kappa=0.0, episodes=6000,
                  alpha=DEFAULT_ALPHA, gamma=0.95, eps_start=1.0,
                  eps_end=0.05, eps_decay=2500, seed=0):
    """w-free USFA training: task vector re-sampled every episode.

    ``sampler`` is a callable ``(rng) -> [N, K]`` task matrix. Returns the
    task-agnostic psi [N, S, A, K], trained at the anchor coupling ``kappa``
    (default: the free region). The mixture over tasks converges slower than
    per-task training, so the default budget is 2-3x that of ``train_sf``."""
    rng = np.random.RandomState(seed)
    n = env.n_agents
    K = env.n_features
    psi = np.zeros((n, env.n_states, env.n_actions, K))
    for ep in range(episodes):
        w = np.asarray(sampler(rng), dtype=float)
        eps = max(eps_end, eps_start + (eps_end - eps_start) * ep / max(1, eps_decay))
        _one_episode(env, psi, w, kappa, alpha, gamma, eps, rng)
    return psi


# --------------------------------------------------------------------- #
# closed-form policies from psi
# --------------------------------------------------------------------- #
def make_l2_policy(psi):
    """Per-agent greedy: a_i = argmax_a psi_i(s_i,a).w_i."""
    n, S, A, K = psi.shape
    grid = int(round(S ** 0.5))

    def policy(positions, w):
        s = np.array([p[0] * grid + p[1] for p in positions])
        q = (psi[range(n), s, :, :] * np.asarray(w)[:, None, :]).sum(-1)
        return q.argmax(-1)
    return policy


def make_l1_policy(psi):
    """Shared-index greedy: all agents take the action a maximizing the sum
    of individual Q's (the synchronized / unconditionally-safe anchor)."""
    n, S, A, K = psi.shape
    grid = int(round(S ** 0.5))

    def policy(positions, w):
        s = np.array([p[0] * grid + p[1] for p in positions])
        q = (psi[range(n), s, :, :] * np.asarray(w)[:, None, :]).sum(-1)
        return np.full(n, int(q.sum(0).argmax()))
    return policy
