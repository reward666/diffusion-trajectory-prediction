from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    for groups in [8, 4, 2, 1]:
        if channels % groups == 0:
            return groups
    return 1


class ConditionedResidualBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, conditioning_dim: int, dropout: float = 0.0):
        super().__init__()
        self.conditioning = nn.Linear(conditioning_dim, output_dim)
        self.input_projection = nn.Conv1d(input_dim, output_dim, kernel_size=1) if input_dim != output_dim else nn.Identity()
        self.net = nn.Sequential(
            nn.GroupNorm(_group_count(input_dim), input_dim),
            nn.SiLU(),
            nn.Conv1d(input_dim, output_dim, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(output_dim), output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv1d(output_dim, output_dim, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, conditioning: torch.Tensor) -> torch.Tensor:
        injected = self.conditioning(conditioning).unsqueeze(-1)
        return self.input_projection(x) + self.net(x) + injected


class TemporalUNetDenoiser(nn.Module):
    def __init__(
        self,
        pred_len: int,
        future_dim: int = 2,
        time_dim: int = 128,
        condition_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if pred_len <= 0:
            raise ValueError("pred_len must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.pred_len = pred_len
        self.future_dim = future_dim
        conditioning_dim = time_dim + condition_dim
        widths = [hidden_dim, hidden_dim * 2, hidden_dim * 4]

        self.input_projection = nn.Conv1d(future_dim, widths[0], kernel_size=3, padding=1)
        self.down_blocks = nn.ModuleList(
            [
                self._make_blocks(widths[level], widths[level], conditioning_dim, num_layers, dropout)
                for level in range(len(widths))
            ]
        )
        self.downsamples = nn.ModuleList(
            [
                nn.Conv1d(widths[level], widths[level + 1], kernel_size=4, stride=2, padding=1)
                for level in range(len(widths) - 1)
            ]
        )
        self.middle = self._make_blocks(widths[-1], widths[-1], conditioning_dim, num_layers, dropout)
        self.upsamples = nn.ModuleList(
            [
                nn.Conv1d(widths[level + 1], widths[level], kernel_size=3, padding=1)
                for level in reversed(range(len(widths) - 1))
            ]
        )
        self.up_blocks = nn.ModuleList(
            [
                self._make_blocks(widths[level] * 2, widths[level], conditioning_dim, num_layers, dropout)
                for level in reversed(range(len(widths) - 1))
            ]
        )
        self.output_projection = nn.Sequential(
            nn.GroupNorm(_group_count(widths[0]), widths[0]),
            nn.SiLU(),
            nn.Conv1d(widths[0], future_dim, kernel_size=3, padding=1),
        )

    @staticmethod
    def _make_blocks(
        input_dim: int,
        output_dim: int,
        conditioning_dim: int,
        num_layers: int,
        dropout: float,
    ) -> nn.ModuleList:
        blocks = [ConditionedResidualBlock(input_dim, output_dim, conditioning_dim, dropout)]
        blocks.extend(
            ConditionedResidualBlock(output_dim, output_dim, conditioning_dim, dropout)
            for _ in range(num_layers - 1)
        )
        return nn.ModuleList(blocks)

    @staticmethod
    def _apply_blocks(
        x: torch.Tensor,
        blocks: nn.ModuleList,
        conditioning: torch.Tensor,
    ) -> torch.Tensor:
        for block in blocks:
            x = block(x, conditioning)
        return x

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
        skips = []
        for blocks, downsample in zip(self.down_blocks[:-1], self.downsamples):
            x = self._apply_blocks(x, blocks, conditioning)
            skips.append(x)
            x = downsample(x)
        x = self._apply_blocks(x, self.down_blocks[-1], conditioning)
        x = self._apply_blocks(x, self.middle, conditioning)

        for upsample, blocks, skip in zip(self.upsamples, self.up_blocks, reversed(skips)):
            x = F.interpolate(x, size=skip.shape[-1], mode="linear", align_corners=False)
            x = upsample(x)
            x = self._apply_blocks(torch.cat([x, skip], dim=1), blocks, conditioning)
        return self.output_projection(x).transpose(1, 2)
