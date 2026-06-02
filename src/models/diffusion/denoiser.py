from __future__ import annotations

import torch
from torch import nn


class TemporalResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, conditioning_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conditioning = nn.Linear(conditioning_dim, hidden_dim)
        self.net = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        injected = self.conditioning(conditioning).unsqueeze(-1)
        return x + self.net(x + injected)


class TemporalDenoiser(nn.Module):
    def __init__(
        self,
        pred_len: int,
        future_dim: int = 2,
        time_dim: int = 128,
        condition_dim: int = 128,
        hidden_dim: int = 256,
        num_layers: int = 6,
        dropout: float = 0.0,
    ):
        super().__init__()
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")
        if hidden_dim % 8 != 0:
            raise ValueError("hidden_dim must be divisible by 8 for GroupNorm.")

        self.pred_len = pred_len
        self.future_dim = future_dim
        conditioning_dim = time_dim + condition_dim
        self.input_projection = nn.Conv1d(future_dim, hidden_dim, kernel_size=3, padding=1)
        self.blocks = nn.ModuleList(
            [TemporalResidualBlock(hidden_dim, conditioning_dim, dropout) for _ in range(num_layers)]
        )
        self.output_projection = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv1d(hidden_dim, future_dim, kernel_size=3, padding=1),
        )

    def forward(
        self,
        noisy_future: torch.Tensor,
        time_emb: torch.Tensor,
        condition_emb: torch.Tensor,
    ) -> torch.Tensor:
        if noisy_future.ndim != 3:
            raise ValueError(f"Expected noisy_future shape [batch, pred_len, future_dim], got {tuple(noisy_future.shape)}.")
        if noisy_future.shape[1] != self.pred_len or noisy_future.shape[2] != self.future_dim:
            raise ValueError(
                "Unexpected noisy_future shape. "
                f"Expected [batch, {self.pred_len}, {self.future_dim}], got {tuple(noisy_future.shape)}."
            )
        if time_emb.ndim != 2 or condition_emb.ndim != 2:
            raise ValueError("Expected time_emb and condition_emb shapes [batch, feature_dim].")
        if noisy_future.shape[0] != time_emb.shape[0] or noisy_future.shape[0] != condition_emb.shape[0]:
            raise ValueError("Batch size mismatch between noisy_future, time_emb, and condition_emb.")

        conditioning = torch.cat([time_emb, condition_emb], dim=-1)
        x = self.input_projection(noisy_future.transpose(1, 2))
        for block in self.blocks:
            x = block(x, conditioning)
        return self.output_projection(x).transpose(1, 2)

