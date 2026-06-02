from __future__ import annotations

import torch
from torch import nn


class LearnedTrajectoryScorer(nn.Module):
    def __init__(
        self,
        context_dim: int,
        pred_len: int,
        future_dim: int = 2,
        hidden_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        candidate_dim = pred_len * future_dim
        self.net = nn.Sequential(
            nn.Linear(context_dim + candidate_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, context: torch.Tensor, candidates: torch.Tensor) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(f"Expected context shape [batch, context_dim], got {tuple(context.shape)}.")
        if candidates.ndim != 4:
            raise ValueError(f"Expected candidates shape [batch, k, pred_len, future_dim], got {tuple(candidates.shape)}.")

        batch_size, num_candidates = candidates.shape[:2]
        repeated_context = context[:, None, :].expand(-1, num_candidates, -1)
        candidate_flat = candidates.reshape(batch_size, num_candidates, -1)
        features = torch.cat([repeated_context, candidate_flat], dim=-1)
        return self.net(features).squeeze(-1)

