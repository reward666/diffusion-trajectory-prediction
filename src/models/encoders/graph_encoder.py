from __future__ import annotations

import torch
from torch import nn


class GraphInteractionEncoder(nn.Module):
    def __init__(
        self,
        node_dim: int = 6,
        edge_dim: int = 6,
        node_hidden_dim: int = 128,
        edge_hidden_dim: int = 64,
        output_dim: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.node_encoder = nn.LSTM(
            input_size=node_dim,
            hidden_size=node_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_dim, edge_hidden_dim),
            nn.SiLU(),
            nn.Linear(edge_hidden_dim, edge_hidden_dim),
        )
        self.attention_score = nn.Sequential(
            nn.Linear(node_hidden_dim * 2 + edge_hidden_dim, node_hidden_dim),
            nn.SiLU(),
            nn.Linear(node_hidden_dim, 1),
        )
        self.neighbor_value = nn.Linear(node_hidden_dim + edge_hidden_dim, node_hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(node_hidden_dim * 2, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim),
        )

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        ego_past = batch["ego_past"]
        neighbor_past = batch["neighbor_past"]
        edge_attr = batch["edge_attr"]
        neighbor_mask = batch["neighbor_mask"] > 0.5
        if ego_past.ndim != 3:
            raise ValueError(f"Expected ego_past shape [batch, obs_len, node_dim], got {tuple(ego_past.shape)}.")
        if neighbor_past.ndim != 4:
            raise ValueError(
                "Expected neighbor_past shape [batch, max_neighbors, obs_len, node_dim], "
                f"got {tuple(neighbor_past.shape)}."
            )

        batch_size, max_neighbors, obs_len, node_dim = neighbor_past.shape
        _, (ego_state, _) = self.node_encoder(ego_past)
        ego_context = ego_state[-1]
        flat_neighbors = neighbor_past.reshape(batch_size * max_neighbors, obs_len, node_dim)
        _, (neighbor_state, _) = self.node_encoder(flat_neighbors)
        neighbor_context = neighbor_state[-1].reshape(batch_size, max_neighbors, -1)
        edge_context = self.edge_encoder(edge_attr)

        ego_expanded = ego_context[:, None, :].expand(-1, max_neighbors, -1)
        score_input = torch.cat([ego_expanded, neighbor_context, edge_context], dim=-1)
        logits = self.attention_score(score_input).squeeze(-1)
        logits = logits.masked_fill(~neighbor_mask, -1e9)
        all_missing = ~neighbor_mask.any(dim=1)
        if all_missing.any():
            logits = logits.clone()
            logits[all_missing] = 0.0
        weights = torch.softmax(logits, dim=1)
        values = self.neighbor_value(torch.cat([neighbor_context, edge_context], dim=-1))
        interaction = torch.sum(values * weights.unsqueeze(-1), dim=1)
        interaction = torch.where(all_missing[:, None], torch.zeros_like(interaction), interaction)
        return self.fusion(torch.cat([ego_context, interaction], dim=-1))
