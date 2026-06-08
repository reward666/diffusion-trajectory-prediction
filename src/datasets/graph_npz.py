from __future__ import annotations

from pathlib import Path

import numpy as np

from src.datasets.normalization import positions_to_deltas


class GraphTrajectoryNPZDataset:
    def __init__(
        self,
        npz_path: str | Path,
        relative_xy: bool = True,
        normalize: bool = False,
        stats: dict[str, np.ndarray] | None = None,
        future_representation: str = "position",
    ):
        self.npz_path = Path(npz_path)
        self.chunk_paths = self._find_chunks(self.npz_path)
        self.chunk_lengths = []
        self._cached_chunk_index = -1
        self._cached_data = None
        for chunk_path in self.chunk_paths:
            with np.load(chunk_path, allow_pickle=False) as data:
                self.chunk_lengths.append(int(data["ego_past"].shape[0]))
                if not hasattr(self, "feature_names"):
                    self.feature_names = data["node_feature_names"]
                    self.edge_feature_names = data["edge_feature_names"]
        self.cumulative_lengths = np.cumsum(self.chunk_lengths)
        self.relative_xy = relative_xy
        self.normalize = normalize
        self.stats = stats
        self.future_representation = future_representation
        if self.normalize and self.stats is None:
            raise ValueError("stats must be provided when normalize=True.")
        if self.future_representation not in {"position", "delta"}:
            raise ValueError("future_representation must be 'position' or 'delta'.")

    @staticmethod
    def _find_chunks(path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        if path.is_dir():
            chunks = sorted(path.glob("chunk_*.npz"))
            if chunks:
                return chunks
        raise FileNotFoundError(f"No graph NPZ file or chunk directory found at {path}.")

    def __len__(self) -> int:
        return int(self.cumulative_lengths[-1])

    def _get_raw_item(self, index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.void]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        chunk_index = int(np.searchsorted(self.cumulative_lengths, index, side="right"))
        chunk_start = 0 if chunk_index == 0 else int(self.cumulative_lengths[chunk_index - 1])
        if chunk_index != self._cached_chunk_index:
            with np.load(self.chunk_paths[chunk_index], allow_pickle=False) as data:
                self._cached_data = {
                    "ego_past": data["ego_past"],
                    "neighbor_past": data["neighbor_past"],
                    "edge_attr": data["edge_attr"],
                    "neighbor_mask": data["neighbor_mask"],
                    "future": data["future"],
                    "meta": data["meta"],
                }
            self._cached_chunk_index = chunk_index
        local_index = index - chunk_start
        return (
            self._cached_data["ego_past"][local_index],
            self._cached_data["neighbor_past"][local_index],
            self._cached_data["edge_attr"][local_index],
            self._cached_data["neighbor_mask"][local_index],
            self._cached_data["future"][local_index],
            self._cached_data["meta"][local_index],
        )

    def _prepare(
        self,
        ego_past: np.ndarray,
        neighbor_past: np.ndarray,
        edge_attr: np.ndarray,
        future: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        ego_past = ego_past.astype(np.float32, copy=True)
        neighbor_past = neighbor_past.astype(np.float32, copy=True)
        edge_attr = edge_attr.astype(np.float32, copy=True)
        future = future.astype(np.float32, copy=True)
        origin = ego_past[-1, :2].copy()
        if self.relative_xy:
            ego_past[:, :2] -= origin
            neighbor_past[:, :, :2] -= origin
            future[:, :2] -= origin
        if self.future_representation == "delta":
            future = positions_to_deltas(future[None, :, :2])[0]
        if self.normalize:
            ego_past = (ego_past - self.stats["ego_mean"]) / self.stats["ego_std"]
            neighbor_past = (neighbor_past - self.stats["neighbor_mean"]) / self.stats["neighbor_std"]
            edge_attr = (edge_attr - self.stats["edge_mean"]) / self.stats["edge_std"]
            if self.future_representation == "delta":
                future = (future - self.stats["future_delta_mean"]) / self.stats["future_delta_std"]
            else:
                future = (future - self.stats["future_mean"]) / self.stats["future_std"]
        return ego_past, neighbor_past, edge_attr, future, origin.astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        ego_past, neighbor_past, edge_attr, neighbor_mask, future, meta = self._get_raw_item(index)
        ego_past, neighbor_past, edge_attr, future, origin = self._prepare(
            ego_past, neighbor_past, edge_attr, future
        )
        return {
            "ego_past": ego_past,
            "neighbor_past": neighbor_past,
            "edge_attr": edge_attr,
            "neighbor_mask": neighbor_mask.astype(np.float32, copy=True),
            "future": future,
            "origin": origin,
            "meta": meta,
        }
