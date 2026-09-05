"""PyTorch port of openpi's src/qc/critic.py (JAX/Flax NNX) for VLA-JEPA.

Same architecture/defaults as the openpi critic: an ensemble of `num_qs`
independent Q-MLPs, hidden_dims=(512,512,512,512), LayerNorm, ReLU, scalar
output per head. Consumes a pooled Qwen embodied-action-token embedding
(mean over the token/sequence dim, in place of openpi's pooled JEPA vision
embedding) + raw low-dim state + a flattened action chunk. Does NOT touch
the VLA-JEPA backbone at all -- same design choice as the openpi critic
(never backprop through the frozen base model).
"""

import torch
import torch.nn as nn


class _QMLP(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: tuple[int, ...], layer_norm: bool = True):
        super().__init__()
        dims = [input_dim, *hidden_dims]
        self.layers = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.norms = nn.ModuleList([nn.LayerNorm(d) for d in hidden_dims]) if layer_norm else None
        self.out = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if self.norms is not None:
                x = self.norms[i](x)
            x = torch.relu(x)
        return self.out(x).squeeze(-1)  # [B]


class QChunkCritic(nn.Module):
    """Ensemble of `num_qs` independent Q-MLPs sharing the same input.
    forward() returns [num_qs, B], matching the openpi critic's output
    convention (e.g. `qs.min(dim=0)` for conservative Q-learning)."""

    def __init__(
        self,
        embed_dim: int,
        proprio_dim: int,
        action_dim: int,
        horizon_length: int,
        hidden_dims: tuple[int, ...] = (512, 512, 512, 512),
        num_qs: int = 5,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.num_qs = num_qs
        input_dim = embed_dim + proprio_dim + action_dim * horizon_length
        self.heads = nn.ModuleList([_QMLP(input_dim, hidden_dims, layer_norm) for _ in range(num_qs)])

    def forward(self, embed: torch.Tensor, proprio: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """embed: [B, embed_dim], proprio: [B, proprio_dim],
        action_chunk: [B, horizon_length, action_dim]. Returns [num_qs, B]."""
        flat_actions = action_chunk.reshape(action_chunk.shape[0], -1)
        x = torch.cat([embed, proprio, flat_actions], dim=-1)
        return torch.stack([head(x) for head in self.heads], dim=0)


class DuelingQChunkCritic(nn.Module):
    """Dueling decomposition Q(s,a) = V(s) + A(s,a), motivated by a
    2026-09-05 diagnostic: the monolithic QChunkCritic above converges
    cleanly (TD relative error ~0.11%, q_mean matches target_q_mean almost
    exactly) but its within-timestep spread across best-of-N candidates
    (what actually drives selection) is ~4 orders of magnitude smaller
    than its cross-STATE spread (q_mean~120, cross-state std~3.3, but
    within-timestep candidate spread ~0.0004). The MSE TD loss on a
    monolithic Q has essentially no gradient incentive to resolve that
    tiny action-dependent residual precisely, since it contributes
    negligibly to the loss compared to getting the large state-dependent
    baseline right -- a dueling split doesn't fix this by itself (V and A
    are still just jointly fit to minimize the same total-Q MSE, nothing
    forces A to carry the action-dependent burden specifically) UNLESS
    combined with mean-centering against a set of alternative actions at
    the SAME state (Wang et al. 2016's identifiability trick): once V(s)
    can't distinguish between actions but the CENTERED advantage term can,
    Q's value for a specific action provably can't be explained by V
    alone, forcing real gradient onto A.

    V depends on (embed, proprio) ONLY -- structurally incapable of
    encoding any action-dependent signal, unlike the monolithic critic
    where V and A-like information can be freely mixed within one head.
    A depends on (embed, proprio, action_chunk), same input as the
    monolithic critic's heads.

    forward_v/forward_a return RAW (uncentered) values -- callers must
    apply the mean-centering themselves using whatever set of alternative
    actions at the same state is available at that call site: see
    qc/train_critic_dueling.py (uses the SAME-timestep cached candidates
    as the baseline set for the TD loss's own Q) and qc/actor.py's
    best_of_n_action (uses the live N best-of-N candidates being scored,
    which naturally IS the right baseline set for eval-time selection --
    no extra forward pass needed there since those candidates are already
    being generated/scored anyway).
    """

    def __init__(
        self,
        embed_dim: int,
        proprio_dim: int,
        action_dim: int,
        horizon_length: int,
        hidden_dims: tuple[int, ...] = (512, 512, 512, 512),
        num_qs: int = 5,
        layer_norm: bool = True,
    ):
        super().__init__()
        self.num_qs = num_qs
        v_input_dim = embed_dim + proprio_dim
        a_input_dim = embed_dim + proprio_dim + action_dim * horizon_length
        self.v_heads = nn.ModuleList([_QMLP(v_input_dim, hidden_dims, layer_norm) for _ in range(num_qs)])
        self.a_heads = nn.ModuleList([_QMLP(a_input_dim, hidden_dims, layer_norm) for _ in range(num_qs)])

    def forward_v(self, embed: torch.Tensor, proprio: torch.Tensor) -> torch.Tensor:
        """embed: [B, embed_dim], proprio: [B, proprio_dim]. Returns [num_qs, B]."""
        x = torch.cat([embed, proprio], dim=-1)
        return torch.stack([head(x) for head in self.v_heads], dim=0)

    def forward_a(self, embed: torch.Tensor, proprio: torch.Tensor, action_chunk: torch.Tensor) -> torch.Tensor:
        """Same input contract as QChunkCritic.forward. Returns [num_qs, B]."""
        flat_actions = action_chunk.reshape(action_chunk.shape[0], -1)
        x = torch.cat([embed, proprio, flat_actions], dim=-1)
        return torch.stack([head(x) for head in self.a_heads], dim=0)

    @staticmethod
    def combine(v: torch.Tensor, a: torch.Tensor, a_baseline_candidates: torch.Tensor) -> torch.Tensor:
        """v: [num_qs, B] (broadcastable), a: [num_qs, B] (advantage of the
        query action(s)), a_baseline_candidates: [num_qs, B, N] (advantage
        of N alternative actions at the SAME states as v/a -- the
        identifiability-constraint reference set). Returns dueling
        Q = v + (a - mean_over_N(a_baseline_candidates)), shape [num_qs, B].
        """
        baseline = a_baseline_candidates.mean(dim=-1)  # [num_qs, B]
        return v + (a - baseline)
