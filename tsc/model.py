import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class MLPQNet(nn.Module):
    """Independent per-agent Q-network (the IDDQN default).

    Every intersection is scored from its OWN observation only -- there is no
    cross-agent information flow, exactly the "independent learners" setting
    (IQL/IDQN). Parameters are shared across agents (one network applied to each
    agent's obs), and an optional per-agent identity embedding keeps
    heterogeneous intersections distinguishable while introducing zero coupling.

    Input  state : [B, N, state_dim]
    Output Q     : [B, N, action_dim]
    """

    def __init__(self, state_dim, action_dim, agent_num, hidden_dim=64,
                 use_agent_emb=True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_num = agent_num
        self.use_agent_emb = use_agent_emb

        in_dim = state_dim
        if use_agent_emb:
            self.agent_emb = nn.Embedding(agent_num, hidden_dim)
            in_dim = state_dim + hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim))

    def _augment(self, state):
        """Append each agent's own identity embedding to its own obs.
        state: [B, N, state_dim] -> [B, N, in_dim]. No cross-agent mixing."""
        if self.use_agent_emb:
            emb = self.agent_emb.weight.unsqueeze(0).expand(state.shape[0], -1, -1)
            state = torch.cat([state, emb], dim=-1)
        return state

    def forward(self, state):
        x = self._augment(state)
        return self.net(x)   # [B, N, action_dim]


class GATLayer(nn.Module):
    """Single graph-attention layer (Velickovic et al. 2018), masked to a fixed
    1-hop adjacency.

    Each node aggregates a learned attention-weighted combination of its 1-hop
    neighbours' (and its own) transformed features. Attention logits for
    non-neighbour pairs are masked to -inf before the softmax, so a node only
    ever attends within its neighbourhood -- the "consider 1-hop neighbours"
    coupling the GAT variant is meant to add on top of the independent MLP.

    adj : [N, N] float {0,1} adjacency WITH self-loops (fixed, not learned).
    """

    def __init__(self, in_dim, out_dim, n_heads=4, concat=True, dropout=0.0,
                 leaky_slope=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.concat = concat
        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)
        # Attention: separate source/target scoring vectors per head.
        self.a_src = nn.Parameter(torch.empty(n_heads, out_dim))
        self.a_dst = nn.Parameter(torch.empty(n_heads, out_dim))
        self.leaky = nn.LeakyReLU(leaky_slope)
        self.dropout = nn.Dropout(dropout)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_src)
        nn.init.xavier_uniform_(self.a_dst)

    def forward(self, h, adj):
        # h: [B, N, in_dim]; adj: [N, N] with self-loops.
        B, N, _ = h.shape
        Wh = self.W(h).view(B, N, self.n_heads, self.out_dim)   # [B, N, H, d]
        Wh = Wh.permute(0, 2, 1, 3)                              # [B, H, N, d]
        # e_ij = LeakyReLU(a_src . Wh_i + a_dst . Wh_j)  (additive attention).
        src = (Wh * self.a_src.view(1, self.n_heads, 1, self.out_dim)).sum(-1)  # [B,H,N]
        dst = (Wh * self.a_dst.view(1, self.n_heads, 1, self.out_dim)).sum(-1)  # [B,H,N]
        e = self.leaky(src.unsqueeze(-1) + dst.unsqueeze(-2))    # [B, H, N, N]
        # Mask out non-neighbours before the softmax.
        mask = (adj.unsqueeze(0).unsqueeze(0) > 0)               # [1, 1, N, N]
        e = e.masked_fill(~mask, float('-inf'))
        attn = torch.softmax(e, dim=-1)                          # over neighbours j
        attn = self.dropout(attn)
        out = torch.matmul(attn, Wh)                            # [B, H, N, d]
        out = out.permute(0, 2, 1, 3)                           # [B, N, H, d]
        if self.concat:
            return out.reshape(B, N, self.n_heads * self.out_dim)
        return out.mean(dim=2)                                  # [B, N, d]


class GATQNet(nn.Module):
    """1-hop GAT Q-network variant.

    A per-agent encoder lifts each intersection's observation (+ optional
    identity embedding) into a hidden representation; ONE graph-attention layer
    then mixes each node with its 1-hop neighbours (masked by the fixed road
    adjacency); a per-agent head reads out Q-values. This is the only place
    cross-agent information enters -- exactly a 1-hop coupled Q-function, in
    contrast to the fully independent :class:`MLPQNet`.

    The adjacency is registered as a buffer so it moves with .to(device) and is
    saved/loaded with the network.

    Input  state : [B, N, state_dim]
    Output Q     : [B, N, action_dim]
    """

    def __init__(self, state_dim, action_dim, agent_num, adj, hidden_dim=64,
                 n_heads=4, use_agent_emb=True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.agent_num = agent_num
        self.use_agent_emb = use_agent_emb

        in_dim = state_dim
        if use_agent_emb:
            self.agent_emb = nn.Embedding(agent_num, hidden_dim)
            in_dim = state_dim + hidden_dim

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU())
        # 1-hop attention mixing. Multi-head, concat -> hidden_dim, then a
        # residual add keeps each node's own encoding available even if a node
        # is isolated.
        assert hidden_dim % n_heads == 0, "hidden_dim must be divisible by n_heads"
        self.gat = GATLayer(hidden_dim, hidden_dim // n_heads, n_heads=n_heads,
                            concat=True)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim))

        adj = torch.as_tensor(adj, dtype=torch.float32)
        self.register_buffer('adj', adj)   # [N, N] with self-loops

    def _augment(self, state):
        if self.use_agent_emb:
            emb = self.agent_emb.weight.unsqueeze(0).expand(state.shape[0], -1, -1)
            state = torch.cat([state, emb], dim=-1)
        return state

    def forward(self, state):
        x = self._augment(state)
        h = self.encoder(x)                       # [B, N, hidden]
        h = h + self.gat(h, self.adj)             # 1-hop neighbour mixing (residual)
        h = F.relu(h)
        return self.head(h)                       # [B, N, action_dim]


class SFQNet(nn.Module):
    """Per-agent SUCCESSOR-FEATURE network: predicts psi(s, a) in R^K per action.

    Identical independent-learner backbone to :class:`MLPQNet` (parameters
    shared across agents, own-observation only, optional identity embedding) but
    the head emits ``action_dim * feat_dim`` values reshaped to
    ``[B, N, action_dim, feat_dim]`` -- the expected discounted feature vector
    psi_i(s_i, a). The task value is the closed form ``Q = psi . w`` computed by
    the caller, so ONE trained psi transfers to any task weighting w without
    retraining. Trained w-free (USFA): a fresh w is drawn per episode and the
    SF-TD target uses phi + gamma * psi(s2, a2) (see agent.py).

    Input  state : [B, N, state_dim]
    Output psi   : [B, N, action_dim, feat_dim]
    """

    def __init__(self, state_dim, action_dim, feat_dim, agent_num,
                 hidden_dim=64, use_agent_emb=True):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.feat_dim = feat_dim
        self.agent_num = agent_num
        self.use_agent_emb = use_agent_emb

        in_dim = state_dim
        if use_agent_emb:
            self.agent_emb = nn.Embedding(agent_num, hidden_dim)
            in_dim = state_dim + hidden_dim

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim * feat_dim))

    def _augment(self, state):
        if self.use_agent_emb:
            emb = self.agent_emb.weight.unsqueeze(0).expand(state.shape[0], -1, -1)
            state = torch.cat([state, emb], dim=-1)
        return state

    def forward(self, state):
        B, N, _ = state.shape
        x = self._augment(state)
        psi = self.net(x)                                  # [B, N, A*K]
        return psi.view(B, N, self.action_dim, self.feat_dim)

    def q_values(self, state, w):
        """Closed-form transfer value Q = psi . w. ``w`` is [K] or [B,N,K]."""
        psi = self.forward(state)                          # [B,N,A,K]
        w_t = torch.as_tensor(w, dtype=psi.dtype, device=psi.device)
        if w_t.dim() == 1:
            return torch.einsum('bnak,k->bna', psi, w_t)   # [B,N,A]
        return torch.einsum('bnak,bnk->bna', psi, w_t)     # per-agent w


class NeighborComposer(nn.Module):
    """Neighbor-limited upper-level composer for the PAID region (TSC-scale).

    Warm-starts from the closed-form SF transfer utilities ``u_i = psi_i . w_i``
    and adds a bounded residual computed from a SINGLE 1-hop graph-attention
    round over the real road adjacency -- so global task information reaches a
    node's decision ONLY through its immediate neighbours (never a global
    softmax over all intersections, which is infeasible at 196 agents). This is
    the TSC analogue of the toy 'comm' composer, respecting the locality
    constraint the paper argues for at scale.

        u_i(s, a) = psi_i(s_i, a) . w_i  +  rho * tanh( head( gat_i(context) ) )
        context_j = [ enc(s_j) , w_j ]     (own obs + own task weights)

    The residual head is zero-initialized so training begins exactly at the
    transfer policy. IGM is preserved trivially: the composer outputs per-agent
    utilities and the joint action is per-agent argmax (decentralized
    execution). Positive-mixing / monotone value is inherited from summing
    per-agent utilities (additive team reward).

    Input  state : [B, N, state_dim] ; w : [B, N, K] (per-agent task weights)
    Output u     : [B, N, action_dim]
    """

    RESID_SCALE = 4.0

    def __init__(self, sf_net, adj, feat_dim, hidden_dim=64, n_heads=4,
                 use_agent_emb=True):
        super().__init__()
        self.sf_net = sf_net                     # frozen or fine-tuned SF backbone
        self.feat_dim = feat_dim
        state_dim = sf_net.state_dim
        action_dim = sf_net.action_dim

        ctx_dim = state_dim + feat_dim           # [own obs, own w]
        self.encoder = nn.Sequential(
            nn.Linear(ctx_dim, hidden_dim), nn.ReLU())
        assert hidden_dim % n_heads == 0
        self.gat = GATLayer(hidden_dim, hidden_dim // n_heads, n_heads=n_heads,
                            concat=True)
        # zero-initialized final layer -> residual starts at 0 (transfer policy)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

        adj = torch.as_tensor(adj, dtype=torch.float32)
        self.register_buffer('adj', adj)

    def forward(self, state, w):
        # base transfer utilities u = psi . w  (per-agent w)
        u_base = self.sf_net.q_values(state, w)            # [B,N,A]
        # neighbour-limited residual
        ctx = torch.cat([state, w], dim=-1)                # [B,N,state+K]
        h = self.encoder(ctx)
        h = h + self.gat(h, self.adj)                      # 1-hop mixing
        h = F.relu(h)
        resid = self.RESID_SCALE * torch.tanh(self.head(h))  # [B,N,A]
        return u_base + resid
