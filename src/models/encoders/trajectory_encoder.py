from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class TrajectoryFeatureSpec:
    feature_names: tuple[str, ...]

    def indices(self, names: list[str]) -> list[int]:
        index_map = {name: i for i, name in enumerate(self.feature_names)}
        missing = [name for name in names if name not in index_map]
        if missing:
            raise ValueError(f"Missing required features: {missing}")
        return [index_map[name] for name in names]


class EgoLeaderEncoder(nn.Module):
    def __init__(
        self,
        feature_names: list[str] | tuple[str, ...],
        ego_hidden_dim: int = 128,
        leader_hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
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
        self.leader_feature_names = [
            "space_headway",
            "time_headway",
            "preceding_exists",
        ]
        self.ego_indices = self.feature_spec.indices(self.ego_feature_names)
        self.leader_indices = self.feature_spec.indices(self.leader_feature_names)

        leader_input_dim = len(self.leader_indices)
        self.ego_encoder = nn.GRU(
            input_size=len(self.ego_indices),
            hidden_size=ego_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.leader_encoder = nn.GRU(
            input_size=leader_input_dim,
            hidden_size=leader_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.fusion = nn.Sequential(
            nn.Linear(ego_hidden_dim + leader_hidden_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def build_leader_features(self, past: torch.Tensor) -> torch.Tensor:
        return past[:, :, self.leader_indices]

    def forward(self, past: torch.Tensor) -> torch.Tensor:
        if past.ndim != 3:
            raise ValueError(f"Expected past shape [batch, obs_len, feature_dim], got {tuple(past.shape)}.")

        ego = past[:, :, self.ego_indices]
        leader = self.build_leader_features(past)
        _, ego_state = self.ego_encoder(ego)
        _, leader_state = self.leader_encoder(leader)
        combined = torch.cat([ego_state[-1], leader_state[-1]], dim=-1)
        return self.fusion(combined)
