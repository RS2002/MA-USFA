"""SFWorld -- a small multi-agent grid toy for the SF+GPI paper.

Purpose
-------
Instantiate, in the smallest possible controlled form, the two channels of
coupling that the theory separates (reward coupling via heterogeneous task
weights ``w``, transition coupling via teammate blocking), with a dial
``kappa`` that moves an operating regime between the FREE region (dynamics
factorize, closed-form SF transfer is exact) and the PAID region (dynamics
entangle, transfer fails, a learned composer is required).

Setup
-----
* ``N`` agents move on a ``G x G`` grid; 5 actions (4 moves + stay).
* ``K`` "resource" cells sit at fixed locations. Agent ``i``'s feature vector
  ``phi_i(s')`` is a dense Gaussian bump around each resource cell -- dense,
  smooth features, exactly like the traffic-signal composite reward (queue /
  wait / pressure / speed are dense congestion signals, not one-hots).
* Task ``w`` = an ``[N, K]`` weight matrix. Reward ``r_i = phi_i(s') . w_i``.
  The feature-reward decomposition IS the Successor-Feature decomposition:
  ``Q_i = psi_i . w_i``.

Transition coupling (the dial): blocking conflicts
--------------------------------------------------
Agents move simultaneously. When two agents intend to move into the SAME cell,
a conflict occurs; with probability ``kappa`` the conflict is resolved by
blocking all but one of them (the blocked agent stays in place). Consequences:

* ``kappa = 0`` -- conflicts never block: agents share cells freely and agent
  i's next cell depends only on its own action. Dynamics FACTORIZE exactly.
  This is the FREE region: ``psi`` learned under any task transfers to any
  task (the kernel is task-independent).
* ``kappa > 0`` -- agent i's next cell depends on the OTHER agents' actions
  (a conflict only exists if a teammate also targets the same cell). Whether
  conflicts happen -- and how costly they are -- depends on the joint policy,
  which depends on the task ``w``. This is the PAID region: the kernel each
  agent experiences changes with the task, so a ``psi`` learned at ``kappa=0``
  is stale under coupling (value-validity fails). This is the "teammates are
  the environment" phenomenon made controllable, with ``kappa`` as the sweep
  axis (the T3 experiment).

Notes
-----
* Observations are strictly LOCAL: agent i sees only its own cell. The
  "composer" methods (see composers.py) additionally receive GLOBAL task
  information ``w_all`` and/or learned messages -- which is exactly why they
  can deconflict in the paid region while decentralized greedy cannot.
* The free/paid partition is a property of the (task, regime) pair: aligned or
  heterogeneous preferences at ``kappa=0`` live in the free region; the same
  tasks at ``kappa>0`` are paid when their policies collide.
"""

import numpy as np

_ACTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=int)


class SFWorld:
    def __init__(self, grid=5, n_agents=3, n_features=4, resource_scale=0.6,
                 horizon=40, seed=0, collision_penalty=2.0, wall_gate=None):
        self.grid = grid
        self.n_agents = n_agents
        self.n_features = n_features
        self.n_actions = len(_ACTIONS)
        self.n_states = grid * grid
        self.horizon = horizon
        self.scale = resource_scale
        self.collision_penalty = collision_penalty
        self.rng = np.random.RandomState(seed)

        # optional interior wall with a narrow gate (a geometric bottleneck).
        # wall_gate = gate_row : a vertical wall between columns c=(grid//2)-1 and
        # c=grid//2 blocks all horizontal crossing EXCEPT at row == gate_row.
        # The wall is STATIC geometry (not coupling): at kappa=0 each agent's
        # own-move kernel is still well defined and factorizes, so the free
        # region stays exact. It only forces heterogeneous-goal agents whose
        # optima lie on opposite sides to funnel through the single gate cell.
        self.wall_gate = wall_gate
        if wall_gate is not None:
            self._wall_c = grid // 2            # wall sits left of column _wall_c
            self._gate_row = int(wall_gate)
        else:
            self._wall_c = None
            self._gate_row = None

        g = grid - 1
        if n_features == 4:
            self.resources = np.array([[0, 0], [0, g], [g, 0], [g, g]], dtype=float)
        elif n_features == 5:
            self.resources = np.array([[0, 0], [0, g], [g, 0], [g, g],
                                       [g / 2, g / 2]], dtype=float)
        elif n_features == 3:
            self.resources = np.array([[0, 0], [g, 0], [g / 2, g]], dtype=float)
        else:
            self.resources = self.rng.rand(n_features, 2) * g

        self._feature_cache = {}

    # ------------------------------------------------------------------ #
    # observation / features / rewards
    # ------------------------------------------------------------------ #
    def cell_to_state(self, pos):
        r, c = pos
        return int(r * self.grid + c)

    def state_to_cell(self, s):
        return (s // self.grid, s % self.grid)

    def phi_of_cell(self, cell):
        """Dense feature vector of a grid cell (Gaussian bump per resource)."""
        cell = tuple(cell)
        if cell not in self._feature_cache:
            c = np.asarray(cell, dtype=float)
            d2 = np.sum((self.resources - c) ** 2, axis=1)
            self._feature_cache[cell] = np.exp(-d2 / (2 * self.scale ** 2))
        return self._feature_cache[cell]

    # ------------------------------------------------------------------ #
    # task sampling
    # ------------------------------------------------------------------ #
    def task_free(self, base=None, sigma=0.05, rng=None):
        """Aligned task: all agents share (nearly) the same preference vector,
        with ONE clearly dominant goal so the greedy argmax is well separated
        (flat uniform preferences make the tabular-SF argmax flip on tiny
        approximation error -- a measurement artifact, not a transfer failure).
        At kappa=0 the shared goal costs nothing (no blocking), so closed-form
        transfer is exact regardless of the shared preference."""
        rng = rng or self.rng
        if base is None:
            base = rng.uniform(0.15, 0.5, size=self.n_features)
            base[rng.randint(self.n_features)] = rng.uniform(1.0, 1.4)   # dominant
        w = np.tile(base, (self.n_agents, 1))
        w += rng.randn(self.n_agents, self.n_features) * sigma
        return np.clip(w, 0.1, 1.5)

    def task_paid(self, rng=None):
        """Heterogeneous task WITH goal overlap (the paid-region generator).

        Agents 0 and 1 share the SAME dominant resource (goal 0), so under
        transition coupling their greedy policies collide -- both head to the
        same corner and are knocked aside. Agent 2 prefers a different goal
        (resource 1) and stays out of the conflict. Each conflicted agent has
        a distinct, weaker SECONDARY goal (agent 0 -> resource 2, agent 1 ->
        resource 3): weaker than the shared goal, so the conflict is worth
        risking, but good enough that a coupling-aware policy can fall back to
        it instead of endlessly contesting. That fallback is exactly what a
        policy trained/composed under the coupled kernel discovers and the
        conflict-unaware closed-form transfer does not -- the source of the
        paid-region value gap the composer recovers."""
        rng = rng or self.rng
        w = rng.uniform(0.1, 0.35, size=(self.n_agents, self.n_features))
        dominant = [0, 0, 1]                      # shared goal 0 for agents 0,1
        secondary = [2, 3, 2]                     # agent-specific fallback goals
        for i, d in enumerate(dominant):
            w[i, d] = rng.uniform(1.1, 1.4)
        for i, s in enumerate(secondary):
            w[i, s] = rng.uniform(0.7, 0.95)
        return np.clip(w, 0.1, 1.5)

    # ------------------------------------------------------------------ #
    # dynamics with the coupling dial
    # ------------------------------------------------------------------ #
    def _blocked_by_wall(self, r0, c0, r1, c1):
        """True iff moving from (r0,c0) to (r1,c1) crosses the interior wall.
        The wall lies between columns _wall_c-1 and _wall_c; crossing it is
        allowed only at row == _gate_row."""
        if self._wall_c is None:
            return False
        lo, hi = min(c0, c1), max(c0, c1)
        crosses = (lo == self._wall_c - 1 and hi == self._wall_c)
        if not crosses:
            return False
        return r0 != self._gate_row          # only the gate row may cross

    def intended_cell(self, pos, a):
        r, c = pos + _ACTIONS[a]
        r = int(min(max(r, 0), self.grid - 1))
        c = int(min(max(c, 0), self.grid - 1))
        if self._blocked_by_wall(pos[0], pos[1], r, c):
            return (int(pos[0]), int(pos[1]))
        return (r, c)

    def step_intended(self, positions, intended, kappa, rng=None):
        """Apply the blocking coupling to the intended actions.

        A conflict (two agents target the same cell) is resolved with
        probability ``kappa`` by blocking all but the highest index: the
        blocked agent is KNOCKED ASIDE to a random adjacent cell instead of
        just holding its position -- so parking on a contested goal is
        impossible and a colliding pair wastes steps re-approaching. This is
        what makes coupling genuinely costly (the paid region): the cost is
        real travel time, not a cosmetic block.

        Returns (new_positions, blocked_mask): positions after the step;
        blocked_mask[i]=1 iff agent i was blocked by a conflict.
        """
        rng = rng or self.rng
        N = self.n_agents
        executed = np.array(intended, dtype=int).copy()
        blocked = np.zeros(N, dtype=float)
        if kappa > 0:
            targets = [self.intended_cell(positions[i], executed[i]) for i in range(N)]
            # agents sharing a target cell: all but the highest index are blocked
            # (with probability kappa per conflict)
            for t in set(targets):
                group = [i for i in range(N) if targets[i] == t]
                if len(group) > 1 and rng.rand() < kappa:
                    for i in group[:-1]:
                        executed[i] = 4          # stay
                        blocked[i] = 1.0
        new_pos = [self._step_pos(positions[i], executed[i]) for i in range(N)]
        if kappa > 0:
            g = self.grid - 1
            for i in range(N):
                if blocked[i]:
                    r, c = positions[i]
                    cand = [(max(0, r - 1), c), (min(g, r + 1), c),
                            (r, max(0, c - 1)), (r, min(g, c + 1))]
                    cand = [x for x in cand
                            if not self._blocked_by_wall(r, c, x[0], x[1])]
                    new_pos[i] = cand[rng.randint(len(cand))]
        return new_pos, blocked

    def _step_pos(self, pos, a):
        r, c = pos + _ACTIONS[a]
        r = int(min(max(r, 0), self.grid - 1))
        c = int(min(max(c, 0), self.grid - 1))
        if self._blocked_by_wall(pos[0], pos[1], r, c):
            return (int(pos[0]), int(pos[1]))
        return (r, c)

    # ------------------------------------------------------------------ #
    # rollout / evaluation
    # ------------------------------------------------------------------ #
    def _ep_rng(self, kappa):
        """Per-(call, kappa) RNG: different regimes get different episode
        draws, so single-seed artifacts (a lucky conflict-free trajectory at
        one kappa) do not masquerade as structure."""
        return np.random.RandomState(self.rng.randint(0, 2 ** 31) ^
                                     int(round(kappa * 1000 + 0.5)) ^
                                     int(self.rng.randint(0, 2 ** 31)))

    def evaluate(self, policy, w, kappa, n_episodes=50, gamma=0.95):
        """Mean discounted joint return of a policy under task (w, kappa).

        Reward = harvest (phi . w) MINUS a collision penalty charged to every
        agent blocked in a conflict. The penalty is the coupling cost that
        makes the paid region genuinely expensive: it lives OUTSIDE the
        successor features (phi is harvest-only), so the closed-form SF value
        Q = psi . w is structurally blind to it -- transfer keeps colliding.

        policy: callable (positions, w) -> [N] intended actions
        """
        ep_rng = self._ep_rng(kappa)
        pen = self.collision_penalty
        rets = np.empty(n_episodes)
        for ep in range(n_episodes):
            cells = ep_rng.choice(self.n_states, self.n_agents, replace=False)
            positions = [self.state_to_cell(c) for c in cells]
            G, disc = 0.0, 1.0
            for t in range(self.horizon):
                intended = np.array(policy(positions, w), dtype=int)
                positions, blocked = self.step_intended(positions, intended, kappa, rng=ep_rng)
                for i in range(self.n_agents):
                    G += disc * (float(self.phi_of_cell(positions[i]) @ w[i])
                                 - pen * blocked[i])
                disc *= gamma
            rets[ep] = G
        return float(rets.mean()), float(rets.std() / np.sqrt(n_episodes))

    # ------------------------------------------------------------------ #
    # diagnostics for the gating rule
    # ------------------------------------------------------------------ #
    def measure_conflicts(self, policy, w, kappa, n_steps=400):
        """Blocking rate under the composed policy: fraction of agent-steps
        in which the agent was blocked by a conflict. ~0 <=> factorized
        dynamics (free region); >0 <=> coupled regime (paid region)."""
        rng = self._ep_rng(kappa)
        cells = rng.choice(self.n_states, self.n_agents, replace=False)
        positions = [self.state_to_cell(c) for c in cells]
        hits, total = 0, 0
        for t in range(n_steps):
            intended = np.array(policy(positions, w), dtype=int)
            positions, blocked = self.step_intended(positions, intended, kappa, rng=rng)
            hits += int(blocked.sum())
            total += self.n_agents
        return hits / total

    def conflict_rate(self, w, kappa, n_steps=400):
        """Blocking rate of a RANDOM policy -- the coupling ceiling for a
        given (w, kappa): how often would uniformly random moves collide."""
        rng = self._ep_rng(kappa)
        cells = rng.choice(self.n_states, self.n_agents, replace=False)
        positions = [self.state_to_cell(c) for c in cells]
        hits, total = 0, 0
        for t in range(n_steps):
            intended = rng.randint(0, self.n_actions, size=self.n_agents)
            positions, blocked = self.step_intended(positions, intended, kappa, rng=rng)
            hits += int(blocked.sum())
            total += self.n_agents
        return hits / total

    # ------------------------------------------------------------------ #
    # exact per-agent optimum (for the free region)
    # ------------------------------------------------------------------ #
    def decoupled_optimal_policy(self, w, gamma=0.95):
        """Value-iteration on the *factorized* kernel (own moves only) -- the
        joint optimum when dynamics decouple. Returns [N] Q-tables."""
        q_tables = []
        for i in range(self.n_agents):
            V = np.zeros(self.n_states)
            for _ in range(500):
                Vnew = np.zeros(self.n_states)
                for s in range(self.n_states):
                    r, c = self.state_to_cell(s)
                    vals = []
                    for a in range(self.n_actions):
                        nr, nc = self._step_pos((r, c), a)
                        s2 = nr * self.grid + nc
                        vals.append(self.phi_of_cell((nr, nc)) @ w[i] + gamma * V[s2])
                    Vnew[s] = max(vals)
                if np.abs(Vnew - V).max() < 1e-7:
                    V = Vnew
                    break
                V = Vnew
            Q = np.zeros((self.n_states, self.n_actions))
            for s in range(self.n_states):
                r, c = self.state_to_cell(s)
                for a in range(self.n_actions):
                    nr, nc = self._step_pos((r, c), a)
                    s2 = nr * self.grid + nc
                    Q[s, a] = self.phi_of_cell((nr, nc)) @ w[i] + gamma * V[s2]
            q_tables.append(Q)
        return q_tables

    def vi_policy_value(self, w, kappa, n_episodes=50, gamma=0.95):
        """Value of the decoupled-optimal policy evaluated in the real env."""
        q_tables = self.decoupled_optimal_policy(w, gamma=gamma)
        return self.evaluate(lambda pos, ww: [np.argmax(q_tables[i][self.cell_to_state(pos[i])])
                                              for i in range(self.n_agents)],
                             w, kappa, n_episodes=n_episodes, gamma=gamma)
