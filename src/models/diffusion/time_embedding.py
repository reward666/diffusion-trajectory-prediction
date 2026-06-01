from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embedding_dim: int):
        super().__init__()
        if embedding_dim < 2:
            raise ValueError("embedding_dim must be at least 2.")
        self.embedding_dim = embedding_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        if timesteps.ndim != 1:
            timesteps = timesteps.reshape(-1)

        half_dim = self.embedding_dim // 2
        device = timesteps.device
        dtype = torch.float32
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        frequencies = torch.exp(torch.arange(half_dim, device=device, dtype=dtype) * -scale)
        angles = timesteps.to(dtype).unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)

        if self.embedding_dim % 2 == 1:
            padding = torch.zeros((embedding.shape[0], 1), device=device, dtype=dtype)
            embedding = torch.cat([embedding, padding], dim=1)
        return embedding


class TimeEmbeddingMLP(nn.Module):
    def __init__(self, embedding_dim: int = 128, hidden_dim: int = 256, output_dim: int = 128):
        super().__init__()
        self.sinusoidal = SinusoidalTimeEmbedding(embedding_dim)
        self.net = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return self.net(self.sinusoidal(timesteps))

