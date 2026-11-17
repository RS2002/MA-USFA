"""Exact N=2 SFWorld kernel, copied semantically from the archived sfworld.py.

State is ordered joint cell index c0*25+c1; action is a0*5+a1.
The last feature is the *positive* blocked indicator, priced by -2.
Time is external: elapsed t=0,...,H-1 and terminal bootstrap is zero.
"""
from __future__ import annotations
import numpy as np

ACTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1], [0, 0]], dtype=np.int64)


class SFWorldExact:
    def __init__(self, grid=5, horizon=40, gamma=.95, kappa=1., seed=0,
                 resource_scale=.6, collision_penalty=2.):
        if grid != 5 or not 0 <= kappa <= 1:
            raise ValueError('This audited implementation supports grid=5, 0<=kappa<=1.')
        self.grid, self.horizon, self.gamma, self.kappa = grid, horizon, gamma, float(kappa)
        self.N, self.F, self.n_agents, self.n_features = 2, 5, 2, 4
        self.n_states, self.n_actions = grid**2, 5  # old names refer to LOCAL sizes
        self.S, self.A = self.n_states**2, 25
        self.collision_penalty = float(collision_penalty)
        self.rng = np.random.RandomState(seed)
        self.joint_actions = np.array([(a, b) for a in range(5) for b in range(5)])
        self.cells = np.array([(r, c) for r in range(grid) for c in range(grid)])
        resources = np.array([[0, 0], [0, grid-1], [grid-1, 0], [grid-1, grid-1]])
        self.harvest = np.exp(-np.sum((self.cells[:, None]-resources[None])**2, -1)
                              / (2*resource_scale**2))
        moved = np.clip(self.cells[:, None] + ACTIONS[None], 0, grid-1)
        self.own_next = moved[..., 0]*grid + moved[..., 1]
        self.state_cells = np.array([(a, b) for a in range(25) for b in range(25)])
        self.initial = (self.state_cells[:, 0] != self.state_cells[:, 1]).astype(float)
        self.initial /= self.initial.sum()
        self._build_kernel()

    def positions(self, s):
        return self.cells[self.state_cells[int(s)]].copy()

    def cell_states(self, s):
        return self.state_cells[int(s)].copy()

    def encode_positions(self, positions):
        p = np.asarray(positions, dtype=int)
        cells = p[:, 0]*self.grid+p[:, 1]
        return int(cells[0]*self.n_states+cells[1])

    def reset(self, rng=None):
        rng = self.rng if rng is None else rng
        c = rng.choice(self.n_states, 2, replace=False)
        return int(c[0]*self.n_states+c[1])

    def action_index(self, actions):
        a = np.asarray(actions)
        return int(a) if a.ndim == 0 else int(a[0]*5+a[1])

    def theta(self, w):
        w = np.asarray(w, dtype=float)
        if w.shape != (2, 4):
            raise ValueError('W must have shape [2,4].')
        return np.concatenate([w, np.full((2, 1), -self.collision_penalty)], -1)

    def reward(self, features, w):
        return np.sum(np.asarray(features)*self.theta(w), axis=(-2, -1))

    def _build_kernel(self):
        # Branch 0 = no block; branches 1..4 retain duplicated boundary candidates.
        self.next_state = np.zeros((self.S, self.A, 5), dtype=np.int64)
        self.prob = np.zeros((self.S, self.A, 5), dtype=float)
        self.branch_features = np.zeros((self.S, self.A, 5, 2, 5), dtype=float)
        for s, (c0, c1) in enumerate(self.state_cells):
            for a, (a0, a1) in enumerate(self.joint_actions):
                n0, n1 = self.own_next[c0, a0], self.own_next[c1, a1]
                self.next_state[s, a, 0] = n0*25+n1
                self.prob[s, a, 0] = 1.
                self.branch_features[s, a, 0, :, :4] = self.harvest[[n0, n1]]
                if n0 == n1 and self.kappa > 0:
                    self.prob[s, a, 0] = 1-self.kappa
                    r, c = self.cells[c0]
                    # Knocked from ORIGINAL position, not from target/stay cell.
                    cand = [(max(0, r-1), c), (min(4, r+1), c),
                            (r, max(0, c-1)), (r, min(4, c+1))]
                    for b, (rr, cc) in enumerate(cand, 1):
                        knocked = rr*5+cc
                        self.next_state[s, a, b] = knocked*25+n1
                        self.prob[s, a, b] = self.kappa/4
                        self.branch_features[s, a, b, :, :4] = self.harvest[[knocked, n1]]
                        self.branch_features[s, a, b, 0, 4] = 1.
        self.expected_features = np.sum(self.prob[..., None, None]*self.branch_features, 2)

    def step(self, s, actions, rng=None):
        rng = self.rng if rng is None else rng
        a = self.action_index(actions)
        b = int(rng.choice(5, p=self.prob[int(s), a]))
        return (int(self.next_state[int(s), a, b]),
                self.branch_features[int(s), a, b].copy(), False)

    def expected_next(self, values):
        """P V for V[S,...], returning [S,A,...]."""
        trailing = (None,)*(np.asarray(values).ndim-1)
        return np.sum(self.prob[(...,)+trailing]*values[self.next_state], axis=2)
