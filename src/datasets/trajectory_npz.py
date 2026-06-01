from __future__ import annotations

from pathlib import Path

import numpy as np


class TrajectoryNPZDataset:
    def __init__(
        self,
        npz_path: str | Path,
        relative_xy: bool = True,
        normalize: bool = False,
        stats: dict[str, np.ndarray] | None = None,
    ):
        self.npz_path = Path(npz_path)
        self.chunk_paths = self._find_chunks(self.npz_path)
        self.chunk_lengths = []
        self._cached_chunk_index = -1
        self._cached_data = None
        for chunk_path in self.chunk_paths:
            data = np.load(chunk_path, allow_pickle=False)
            self.chunk_lengths.append(int(data["past"].shape[0]))
            if not hasattr(self, "feature_names"):
                self.feature_names = data["feature_names"]
        self.cumulative_lengths = np.cumsum(self.chunk_lengths)
        self.relative_xy = relative_xy
        self.normalize = normalize
        self.stats = stats

        if self.normalize and self.stats is None:
            raise ValueError("stats must be provided when normalize=True.")

    @staticmethod
    def _find_chunks(path: Path) -> list[Path]:
        if path.is_file():
            return [path]
        if path.is_dir():
            chunks = sorted(path.glob("chunk_*.npz"))
            if chunks:
                return chunks
        raise FileNotFoundError(f"No trajectory NPZ file or chunk directory found at {path}.")

    def __len__(self) -> int:
        return int(self.cumulative_lengths[-1])

    def _get_raw_item(self, index: int) -> tuple[np.ndarray, np.ndarray, np.void]:
        if index < 0:
            index += len(self)
        if index < 0 or index >= len(self):
            raise IndexError(index)
        chunk_index = int(np.searchsorted(self.cumulative_lengths, index, side="right"))
        chunk_start = 0 if chunk_index == 0 else int(self.cumulative_lengths[chunk_index - 1])
        if chunk_index != self._cached_chunk_index:
            with np.load(self.chunk_paths[chunk_index], allow_pickle=False) as data:
                self._cached_data = {
                    "past": data["past"],
                    "future": data["future"],
                    "meta": data["meta"],
                }
            self._cached_chunk_index = chunk_index
        local_index = index - chunk_start
        return (
            self._cached_data["past"][local_index],
            self._cached_data["future"][local_index],
            self._cached_data["meta"][local_index],
        )

    def _prepare(self, past: np.ndarray, future: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        past = past.astype(np.float32, copy=True)
        future = future.astype(np.float32, copy=True)
        origin = past[-1, :2].copy()

        if self.relative_xy:
            past[:, :2] -= origin
            future[:, :2] -= origin

        if self.normalize:
            past = (past - self.stats["past_mean"]) / self.stats["past_std"]
            future = (future - self.stats["future_mean"]) / self.stats["future_std"]

        return past, future, origin.astype(np.float32)

    def __getitem__(self, index: int) -> dict:
        raw_past, raw_future, meta = self._get_raw_item(index)
        past, future, origin = self._prepare(raw_past, raw_future)
        return {
            "past": past,
            "future": future,
            "origin": origin,
            "meta": meta,
        }
