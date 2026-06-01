from __future__ import annotations

import torch
from torch import nn


class MLPDenoiser(nn.Module):
    def __init__(
        self,
        pred_len: int,
        future_dim: int = 2,
        time_dim: int = 128,
        condition_dim: int = 128,
        hidden_dim: int = 512,
        num_layers: int = 4,
        dropout: float = 0.0,
    ):
        super().__init__()
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if num_layers < 2:
            raise ValueError("num_layers must be at least 2.")

        self.pred_len = pred_len
        self.future_dim = future_dim
        input_dim = pred_len * future_dim + time_dim + condition_dim
        output_dim = pred_len * future_dim

        layers: list[nn.Module] = []
        current_dim = input_dim
        for _ in range(num_layers - 1):
            layers.extend(
                [
                    nn.Linear(current_dim, hidden_dim),
                    nn.SiLU(),
                ]
            )
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))
        self.net = nn.Sequential(*layers)

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
        if time_emb.ndim != 2:
            raise ValueError(f"Expected time_emb shape [batch, time_dim], got {tuple(time_emb.shape)}.")
        if condition_emb.ndim != 2:
            raise ValueError(f"Expected condition_emb shape [batch, condition_dim], got {tuple(condition_emb.shape)}.")
        if noisy_future.shape[0] != time_emb.shape[0] or noisy_future.shape[0] != condition_emb.shape[0]:
            raise ValueError("Batch size mismatch between noisy_future, time_emb, and condition_emb.")

        batch_size = noisy_future.shape[0]
        noisy_flat = noisy_future.reshape(batch_size, self.pred_len * self.future_dim)
        model_input = torch.cat([noisy_flat, time_emb, condition_emb], dim=-1)
        pred_noise = self.net(model_input)
        return pred_noise.reshape(batch_size, self.pred_len, self.future_dim)

