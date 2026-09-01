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
