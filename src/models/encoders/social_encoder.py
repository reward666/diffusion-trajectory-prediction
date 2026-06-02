from __future__ import annotations

import torch
from torch import nn

from src.models.encoders.trajectory_encoder import TrajectoryFeatureSpec
from src.preprocessing.ngsim_schema import SOCIAL_NEIGHBOR_ATTRIBUTES, SOCIAL_NEIGHBOR_SLOTS


class TemporalAttentionPool(nn.Module):
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        logits = self.score(sequence).squeeze(-1)
        if mask is not None:
            logits = logits.masked_fill(~mask, -1e9)
            all_missing = ~mask.any(dim=1)
            if all_missing.any():
                logits = logits.clone()
                logits[all_missing] = 0.0
        weights = torch.softmax(logits, dim=1)
        return torch.sum(sequence * weights.unsqueeze(-1), dim=1)


class EgoSocialAttentionEncoder(nn.Module):
    def __init__(
        self,
        feature_names: list[str] | tuple[str, ...],
        ego_hidden_dim: int = 128,
        neighbor_hidden_dim: int = 64,
        output_dim: int = 128,
        num_attention_heads: int = 4,
        dropout: float = 0.0,
        neighbor_exists_thresholds: list[float] | tuple[float, ...] | None = None,
    ):
        super().__init__()
        self.feature_spec = TrajectoryFeatureSpec(tuple(feature_names))
        self.ego_feature_names = [
            "x",
            "y",
            "vx",
            "vy",
            "speed",
            "acc",
            "lane_id",
            "vehicle_class",
            "length",
            "width",
        ]
        self.neighbor_slots = SOCIAL_NEIGHBOR_SLOTS
        self.neighbor_attributes = SOCIAL_NEIGHBOR_ATTRIBUTES
        self.ego_indices = self.feature_spec.indices(self.ego_feature_names)
        self.neighbor_indices = [
            self.feature_spec.indices([f"{slot}_{attribute}" for attribute in self.neighbor_attributes])
            for slot in self.neighbor_slots
        ]
        self.exists_attribute_index = self.neighbor_attributes.index("exists")
        if neighbor_exists_thresholds is None:
            neighbor_exists_thresholds = [0.0] * len(self.neighbor_slots)
        if len(neighbor_exists_thresholds) != len(self.neighbor_slots):
            raise ValueError(
                f"Expected {len(self.neighbor_slots)} neighbor exists thresholds, "
                f"got {len(neighbor_exists_thresholds)}."
            )
        self.register_buffer(
            "neighbor_exists_thresholds",
            torch.tensor(neighbor_exists_thresholds, dtype=torch.float32),
        )

        self.ego_gru = nn.GRU(len(self.ego_indices), ego_hidden_dim, batch_first=True)
        self.neighbor_gru = nn.GRU(len(self.neighbor_attributes), neighbor_hidden_dim, batch_first=True)
        self.ego_temporal_pool = TemporalAttentionPool(ego_hidden_dim)
        self.neighbor_temporal_pool = TemporalAttentionPool(neighbor_hidden_dim)
        self.ego_query = nn.Linear(ego_hidden_dim, neighbor_hidden_dim)
        self.social_attention = nn.MultiheadAttention(
            embed_dim=neighbor_hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.fusion = nn.Sequential(
            nn.Linear(ego_hidden_dim + neighbor_hidden_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        if past.ndim != 3:
            raise ValueError(f"Expected past shape [batch, obs_len, feature_dim], got {tuple(past.shape)}.")

        batch_size, obs_len = past.shape[:2]
        ego_sequence, _ = self.ego_gru(past[:, :, self.ego_indices])
        ego_context = self.ego_temporal_pool(ego_sequence)

        neighbors = torch.stack([past[:, :, indices] for indices in self.neighbor_indices], dim=1)
        neighbor_mask = (
            neighbors[:, :, :, self.exists_attribute_index]
            > self.neighbor_exists_thresholds[None, :, None]
        )
        flattened_neighbors = neighbors.reshape(batch_size * len(self.neighbor_slots), obs_len, -1)
        flattened_mask = neighbor_mask.reshape(batch_size * len(self.neighbor_slots), obs_len)
        neighbor_sequence, _ = self.neighbor_gru(flattened_neighbors)
        neighbor_context = self.neighbor_temporal_pool(neighbor_sequence, flattened_mask)
        neighbor_context = neighbor_context.reshape(batch_size, len(self.neighbor_slots), -1)

        slot_missing = ~neighbor_mask.any(dim=2)
        all_slots_missing = slot_missing.all(dim=1)
        if all_slots_missing.any():
            neighbor_context = neighbor_context.clone()
            slot_missing = slot_missing.clone()
            neighbor_context[all_slots_missing, 0] = 0.0
            slot_missing[all_slots_missing, 0] = False

        query = self.ego_query(ego_context).unsqueeze(1)
        social_context, _ = self.social_attention(
            query=query,
            key=neighbor_context,
            value=neighbor_context,
            key_padding_mask=slot_missing,
            need_weights=False,
        )
        return self.fusion(torch.cat([ego_context, social_context.squeeze(1)], dim=-1))
