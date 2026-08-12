"""Learned policy composers for the toy (numpy only).

The PAID-region mechanism: when closed-form SF transfer is no longer valid
(transition coupling > 0), a *learned upper-level composer* is trained on the
task. All three flavors share the same joint value

    Q(s,a) = sum_i lam_i(s, w_all) * u_i(s, a_i),   lam_i = N * softmax(score),

with positive mixing weights so the joint argmax factorizes into per-agent
argmax (IGM). They differ ONLY in how global task information enters each
agent's *utility* (the decision), which is exactly the axis the paper studies:

* ``qmix`` -- ``u_i(s,a) = table_i(s_i,a)`` (transfer-initialized). Global
  ``w`` enters only the mixing weights, never the utilities, so extraction is
  per-agent argmax of purely local tables: a QMIX composer CANNOT deconflict.
  Prediction: qmix ~ per-task retrain in the paid region.
* ``attn`` -- ``u_i = table_i + rho * tanh(head(attention_i(contexts)))``,
  where each agent attends (scaled dot-product) over all agents' contexts
  ``c_j = [onehot(s_j), w_j]``. Global task + state information reaches the
  decision, so agent i can defer on a contested goal it values less.
* ``comm`` -- ``u_i = table_i + rho * tanh(head(s_i, w_i, m_i))`` with ``m_i``
  the output of ``T`` rounds of (tanh-bounded) message passing over the
  complete graph. Messages carry *live* state, enabling dynamic deconfliction.

Training: joint-TD on the *global* reward, semi-gradient, with a hard-synced
TARGET copy of every component (table, mixing net, and all residual /
attention / message nets) used to form the bootstrap target -- without target
copies of the residual nets the bootstrap chases the online residual and
diverges. Residuals are zero-initialized and ``tanh``-bounded (scale ``rho``)
so training begins exactly at the transfer policy and stays stable.
"""

import numpy as np


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


class MLP:
    """Tiny 2-layer ReLU net with hand-rolled backprop (numpy)."""

    def __init__(self, in_d, hid, out_d, seed=0, zero_final=False):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(in_d, hid) * np.sqrt(2.0 / in_d)
        self.b1 = np.zeros(hid)
        if zero_final:
            self.W2 = np.zeros((hid, out_d))
            self.b2 = np.zeros(out_d)
        else:
            self.W2 = rng.randn(hid, out_d) * np.sqrt(2.0 / hid)
            self.b2 = np.zeros(out_d)

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        h = np.maximum(z1, 0.0)
        return h @ self.W2 + self.b2, (X, h)

    def backward(self, dout, cache):
        X, h = cache
        dW2 = h.T @ dout
        db2 = dout.sum(0)
        dh = dout @ self.W2.T
        dz1 = dh * (h > 0)
        dW1 = X.T @ dz1
        db1 = dz1.sum(0)
        dX = dz1 @ self.W1.T
        return (dW1, db1, dW2, db2), dX

    def sgd(self, grads, lr, clip=5.0):
        dW1, db1, dW2, db2 = grads
        if clip is not None:
            dW1 = np.clip(dW1, -clip, clip); dW2 = np.clip(dW2, -clip, clip)
            db1 = np.clip(db1, -clip, clip); db2 = np.clip(db2, -clip, clip)
        self.W1 -= lr * dW1; self.b1 -= lr * db1
        self.W2 -= lr * dW2; self.b2 -= lr * db2

    def clone(self):
        c = MLP.__new__(MLP)
        c.W1 = self.W1.copy(); c.b1 = self.b1.copy()
        c.W2 = self.W2.copy(); c.b2 = self.b2.copy()
        return c

    def copy_from(self, other):
        self.W1[:] = other.W1; self.b1[:] = other.b1
        self.W2[:] = other.W2; self.b2[:] = other.b2


RESID_SCALE = 4.0          # bound on the learned utility correction


class Composer:
    FLAVORS = ("qmix", "attn", "comm")

    def __init__(self, env, psi, w_task, flavor="qmix", hidden=32,
                 msg_dim=16, attn_dim=16, n_rounds=2, gamma=0.95, seed=0):
        assert flavor in self.FLAVORS
        self.env = env
        self.flavor = flavor
        self.gamma = gamma
        self.n_agents = N = env.n_agents
        self.n_states = env.n_states
        self.n_actions = env.n_actions
        self.K = env.n_features
        self.msg_dim = msg_dim
        self.attn_dim = attn_dim
        self.n_rounds = n_rounds
        d = self.n_states + self.K                       # per-agent context dim

        # ---- tabular utilities, transfer-initialized ----
        w = np.asarray(w_task, dtype=float)
        self.u = np.zeros((N, self.n_states, self.n_actions))
        for i in range(N):
            self.u[i] = psi[i] @ w[i]
        self.u_target = self.u.copy()
        self.w_all = w.reshape(-1)

        # ---- monotone mixing net (shared by all flavors) ----
        # zero-initialized output: scores = 0 -> lam = N * softmax(0) = 1 for
        # every agent. For additive joint rewards (R = sum_i r_i) the correct
        # decomposition IS lam_i = 1, so the composer starts exactly at the
        # additive decomposition and learns any deviation from there.
        self.lam_net = MLP(N * self.n_states + N * self.K, hidden, N,
                           seed=seed, zero_final=True)
        self.lam_tgt = self.lam_net.clone()

        # ---- attn: scaled dot-product self-attention over contexts ----
        r2 = np.random.RandomState(seed + 2)
        self.Wq = r2.randn(d, attn_dim) * np.sqrt(1.0 / d)
        self.Wk = r2.randn(d, attn_dim) * np.sqrt(1.0 / d)
        self.Wv = r2.randn(d, attn_dim) * np.sqrt(1.0 / d)
        self.attn_head = MLP(attn_dim, hidden, self.n_actions,
                             seed=seed + 3, zero_final=True)

        # ---- comm: tanh-bounded message passing ----
        self.m0 = MLP(d, hidden, msg_dim, seed=seed + 4)
        self.mg = MLP(2 * msg_dim, hidden, msg_dim, seed=seed + 5)
        self.comm_head = MLP(d + msg_dim, hidden, self.n_actions,
                             seed=seed + 6, zero_final=True)

        # target copies of the residual nets
        self._sync_targets()

    def _sync_targets(self):
        self.u_target = self.u.copy()
        self.lam_tgt.copy_from(self.lam_net)
        self.Wq_t, self.Wk_t, self.Wv_t = self.Wq.copy(), self.Wk.copy(), self.Wv.copy()
        self.attn_head_t = self.attn_head.clone()
        self.m0_t = self.m0.clone(); self.mg_t = self.mg.clone()
        self.comm_head_t = self.comm_head.clone()

    # ------------------------------------------------------------------ #
    # inputs
    # ------------------------------------------------------------------ #
    def _onehot(self, s):
        B = len(s)
        X = np.zeros((B, self.n_states))
        X[np.arange(B), s] = 1.0
        return X

    def _contexts(self, s_batch):
        """[B, N, S+K] per-agent contexts [onehot(s_i), w_i]."""
        B, N = s_batch.shape
        C = np.empty((B, N, self.n_states + self.K))
        for i in range(N):
            C[:, i, :self.n_states] = self._onehot(s_batch[:, i])
            C[:, i, self.n_states:] = self.w_all[self.K * i:self.K * (i + 1)]
        return C

    # ------------------------------------------------------------------ #
    # utilities  (returns (U, cache))
    # ------------------------------------------------------------------ #
    def utilities(self, s_batch, target=False):
        B, N = s_batch.shape
        u_tab = self.u_target if target else self.u
        U = np.stack([u_tab[i][s_batch[:, i]] for i in range(N)], axis=1).copy()
        cache = {}
        if self.flavor == "attn":
            C = self._contexts(s_batch)
            Wq, Wk, Wv = (self.Wq_t, self.Wk_t, self.Wv_t) if target else (self.Wq, self.Wk, self.Wv)
            head = self.attn_head_t if target else self.attn_head
            Q = C @ Wq; Kk = C @ Wk; Vv = C @ Wv               # [B,N,ad]
            scores = np.einsum("bid,bjd->bij", Q, Kk) / np.sqrt(self.attn_dim)
            Aw = _softmax(scores, axis=-1)                     # [B,N,N]
            Z = np.einsum("bij,bjd->bid", Aw, Vv)              # [B,N,ad]
            raw = np.empty((B, N, self.n_actions)); hh = []
            for i in range(N):
                ri, ci = head.forward(Z[:, i]); raw[:, i] = ri; hh.append(ci)
            resid = RESID_SCALE * np.tanh(raw)
            U += resid
            cache = dict(C=C, Q=Q, Kk=Kk, Vv=Vv, Aw=Aw, Z=Z, raw=raw, hh=hh)
        elif self.flavor == "comm":
            C = self._contexts(s_batch)
            m0 = self.m0_t if target else self.m0
            mg = self.mg_t if target else self.mg
            head = self.comm_head_t if target else self.comm_head
            m = np.empty((B, N, self.msg_dim)); c0 = []
            for i in range(N):
                mi, ci = m0.forward(C[:, i]); m[:, i] = np.tanh(mi); c0.append(ci)
            m0_out = m.copy(); cg = []; m_rounds = []
            for _ in range(self.n_rounds):
                ssum = m.sum(1)
                for i in range(N):
                    mi, ci = mg.forward(np.concatenate([m[:, i], ssum], 1))
                    m[:, i] = np.tanh(mi); cg.append(ci)
                m_rounds.append(m.copy())
            raw = np.empty((B, N, self.n_actions)); hh = []
            for i in range(N):
                ri, ci = head.forward(np.concatenate([C[:, i], m[:, i]], 1))
                raw[:, i] = ri; hh.append(ci)
            resid = RESID_SCALE * np.tanh(raw)
            U += resid
            cache = dict(C=C, m=m, m0_out=m0_out, m_rounds=m_rounds,
                         c0=c0, cg=cg, hh=hh, raw=raw)
        return U, cache

    # ------------------------------------------------------------------ #
    # mixing weights  (returns (lam, cache))
    # ------------------------------------------------------------------ #
    def lam(self, s_batch, target=False):
        """lam [B, N] = N * softmax(score): positive (IGM preserved) and
        summing to N (correct joint-value scale)."""
        B, N = s_batch.shape
        flat = np.concatenate([self._onehot(s_batch[:, i]) for i in range(N)] +
                              [np.tile(self.w_all, (B, 1))], axis=1)
        net = self.lam_tgt if target else self.lam_net
        scores, (X, h) = net.forward(flat)
        sm = _softmax(scores)
        return N * sm, dict(X=X, h=h, sm=sm)

    def greedy_actions(self, s):
        U, _ = self.utilities(np.asarray(s)[None, :])
        return U[0].argmax(-1)

    def policy(self):
        grid = int(round(self.n_states ** 0.5))

        def fn(positions, w):
            s = np.array([p[0] * grid + p[1] for p in positions])
            return self.greedy_actions(s)
        return fn

    # PLACEHOLDER_TRAIN
    def train_task(self, w_task, kappa, episodes=1400, alpha_table=5e-3,
                   alpha_net=None, batch_size=64, buffer_size=20000,
                   eps_start=0.2, eps_end=0.02, eps_decay=800,
                   target_sync=100, eval_every=200, upd_every=4, seed=0):
        """Per-agent-TD training of the composer on task (w_task, kappa).

        Warm-starts from the closed-form transfer policy (low initial
        exploration so the good initialization survives). Gradient steps are
        taken every ``upd_every`` env steps; the bootstrap target uses
        hard-synced target copies of EVERY component. ``alpha_net`` defaults
        to 1e-3, lowered to 3e-4 for the ``comm`` flavor whose gradients pass
        through ``T`` message-passing rounds and are noisier.
        Returns (final_policy, eval trace, td trace)."""
        if alpha_net is None:
            alpha_net = 3e-4 if self.flavor == "comm" else 1e-3
        env = self.env
        rng = np.random.RandomState(seed)
        w = np.asarray(w_task, dtype=float)
        self.w_all = w.reshape(-1)
        N = self.n_agents

        buf_s, buf_a, buf_r, buf_s2, buf_d = [], [], [], [], []
        upd_steps, evals, td_trace = 0, [], []

        for ep in range(episodes):
            eps = max(eps_end, eps_start + (eps_end - eps_start) * ep / max(1, eps_decay))
            cells = rng.choice(env.n_states, N, replace=False)
            positions = [env.state_to_cell(c) for c in cells]
            for t in range(env.horizon):
                s = np.array([env.cell_to_state(p) for p in positions])
                U, _ = self.utilities(s[None, :])
                greedy = U[0].argmax(-1)
                explore = rng.rand(N) < eps
                a = np.where(explore, rng.randint(0, self.n_actions, size=N), greedy)

                positions, blocked = env.step_intended(positions, a, kappa, rng=rng)
                pen = getattr(env, "collision_penalty", 0.0)
                r = np.array([float(env.phi_of_cell(positions[i]) @ w[i])
                              - pen * blocked[i]
                              for i in range(N)])                     # [N] per-agent
                s2 = np.array([env.cell_to_state(p) for p in positions])
                done = float(t == env.horizon - 1)

                buf_s.append(s); buf_a.append(a); buf_r.append(r)
                buf_s2.append(s2); buf_d.append(done)
                if len(buf_s) > buffer_size:
                    for b in (buf_s, buf_a, buf_s2):
                        b.pop(0)
                    buf_r.pop(0); buf_d.pop(0)

                if len(buf_s) >= batch_size and (t + 1) % upd_every == 0:
                    idx = rng.randint(0, len(buf_s), size=batch_size)
                    S = np.stack([buf_s[i] for i in idx])
                    A = np.stack([buf_a[i] for i in idx])
                    R = np.stack([buf_r[i] for i in idx])             # [B, N]
                    S2 = np.stack([buf_s2[i] for i in idx])
                    D = np.array([buf_d[i] for i in idx])
                    td = self._td_update(S, A, R, S2, D, alpha_table, alpha_net)
                    td_trace.append(td)
                    upd_steps += 1
                    if upd_steps % target_sync == 0:
                        self._sync_targets()

            if (ep + 1) % eval_every == 0:
                evals.append(env.evaluate(self.policy(), w, kappa,
                                          n_episodes=25, gamma=self.gamma)[0])
        return self.policy(), evals, td_trace

    def _td_update(self, S, A, R, S2, D, alpha_table, alpha_net):
        """One semi-gradient step. Returns mean |delta|.

        Per-agent TD on each agent's OWN reward (honest targets):
            delta_i = r_i + gamma (1-done) max_{a'} u_i(s2, a') - u_i(s, a_i)
        The IGM-max JOINT target (QMIX's update) is NOT used: under
        transition coupling the joint max overestimates the achievable value
        (the team cannot realize sum_i max u_i simultaneously), which drives
        the composed value below the per-agent fixed point (measured in T3).
        With per-agent targets the composer's fixed point is the honest
        marginal value of the composed greedy policy; the global-information
        residual (attn/comm) is what can push beyond the decentralized
        baseline. The target side uses hard-synced target copies.
        """
        B = len(S)
        N = self.n_agents
        U, cache_u = self.utilities(S)
        U2_tab, _ = self.utilities(S2, target=True)
        rows = np.arange(B)[:, None]
        cols = np.arange(N)[None, :]

        u_taken = U[rows, cols, A]                                    # [B, N]
        target = R + self.gamma * (1.0 - D)[:, None] * U2_tab.max(-1)  # [B, N]
        delta = target - u_taken
        G = 2.0 * delta

        # ---- utility table (taken side only): descent u_i += 2 alpha delta_i ----
        for i in range(N):
            np.add.at(self.u, (i, S[:, i], A[:, i]), alpha_table * G[:, i])

        # ---- mixing net still learns the joint value (reporting only) ----
        lam_s, cache_s = self.lam(S)
        q_taken = (lam_s * u_taken).sum(-1)
        q_next = (lam_s * U2_tab.max(-1)).sum(-1)
        lam_delta = (R.sum(-1) + self.gamma * (1.0 - D) * q_next) - q_taken
        dlam_s = -2.0 * lam_delta[:, None] * u_taken                  # [B, N]
        sm = cache_s["sm"]
        dscore = N * sm * (dlam_s - (dlam_s * sm).sum(1, keepdims=True))
        self.lam_net.sgd(self.lam_net.backward(dscore, (cache_s["X"], cache_s["h"]))[0],
                         alpha_net)

        # ---- residual nets (attn / comm): dL/du at taken actions ----
        dU_taken = -G
        if self.flavor == "attn":
            self._attn_residual_backward(S, A, cache_u, dU_taken, alpha_net)
        elif self.flavor == "comm":
            self._comm_backward(S, A, cache_u, dU_taken, alpha_net)
        return float(np.abs(delta).mean())

    # ------------------------------------------------------------------ #
    # attn residual backward
    # ------------------------------------------------------------------ #
    def _attn_residual_backward(self, S, A, cache, dU_taken, lr):
        """dL/d{head, Wq, Wk, Wv}: through tanh-scale -> head -> attention."""
        B, N = S.shape
        C, Q, Kk, Vv, Aw, Z, raw, hh = (cache[k] for k in
                                         ("C", "Q", "Kk", "Vv", "Aw", "Z", "raw", "hh"))
        ad = self.attn_dim

        dZ = np.zeros((B, N, ad)); dhead_out = np.empty_like(raw)
        for i in range(N):
            dout = np.zeros((B, self.n_actions))
            dout[np.arange(B), A[:, i]] = dU_taken[:, i]
            dresid = RESID_SCALE * dout * (1.0 - np.tanh(raw[:, i]) ** 2)
            grads, dXi = self.attn_head.backward(dresid, hh[i])
            self.attn_head.sgd(grads, lr)
            dZ[:, i] = dXi

        # attention: Z = Aw @ Vv ; Aw = softmax(Q Kk^T / sqrt(ad))
        dAw = np.einsum("bid,bjd->bij", dZ, Vv)                      # [B,N,N]
        dVv = np.einsum("bij,bid->bjd", Aw, dZ)                      # [B,N,ad]
        dscore = Aw * (dAw - (dAw * Aw).sum(-1, keepdims=True))
        dQ = np.einsum("bij,bjd->bid", dscore, Kk) / np.sqrt(ad)     # [B,N,ad]
        dKk = np.einsum("bij,bid->bjd", dscore, Q) / np.sqrt(ad)     # [B,N,ad]
        C2 = C.reshape(B * N, -1).T                    # [d, BN]
        self.Wq -= lr * C2 @ dQ.reshape(B * N, -1)
        self.Wk -= lr * C2 @ dKk.reshape(B * N, -1)
        self.Wv -= lr * C2 @ dVv.reshape(B * N, -1)

    # ------------------------------------------------------------------ #
    # comm residual backward
    # ------------------------------------------------------------------ #
    def _comm_backward(self, S, A, cache, dU_taken, lr):
        """dL/d{comm_head, mg, m0} through the tanh-bounded message chain."""
        B, N = S.shape
        C, m_T, m0_out, m_rounds, c0, cg, hh, raw = (cache[k] for k in
            ("C", "m", "m0_out", "m_rounds", "c0", "cg", "hh", "raw"))

        dmg_in = [np.zeros((B, self.msg_dim)) for _ in range(N)]
        for i in range(N):
            dout = np.zeros((B, self.n_actions))
            dout[np.arange(B), A[:, i]] = dU_taken[:, i]
            dresid = RESID_SCALE * dout * (1.0 - np.tanh(raw[:, i]) ** 2)
            grads, dX = self.comm_head.backward(dresid, hh[i])
            self.comm_head.sgd(grads, lr)
            dmg_in[i] = dX[:, self.n_states + self.K:] * (1.0 - m_T[:, i] ** 2)

        for r in range(self.n_rounds - 1, -1, -1):
            m_r = m_rounds[r]
            ssum_in = np.zeros((B, self.msg_dim))
            for i in range(N):
                din = dmg_in[i]
                grads, dX = self.mg.backward(din, cg[r * N + i])
                self.mg.sgd(grads, lr)
                dmg_in[i] = dX[:, :self.msg_dim] * (1.0 - m_r[:, i] ** 2)
                ssum_in += dX[:, self.msg_dim:] * (1.0 - m_r[:, i] ** 2)
            for i in range(N):
                dmg_in[i] = dmg_in[i] + ssum_in
        for i in range(N):
            grads, _ = self.m0.backward(dmg_in[i] * (1.0 - m0_out[:, i] ** 2), c0[i])
            self.m0.sgd(grads, lr)
