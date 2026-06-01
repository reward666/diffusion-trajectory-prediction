from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

from src.datasets.normalization import compute_normalization_stats, load_stats, save_stats
from src.datasets.trajectory_npz import TrajectoryNPZDataset


@dataclass(frozen=True)
class DataConfig:
    split_dir: Path = Path("data/splits")
    prefix: str = "ngsim"
    batch_size: int = 64
    shuffle_train: bool = True
    seed: int = 42
    relative_xy: bool = True
    normalize: bool = True
    stats_path: Path = Path("data/splits/ngsim_stats.npz")


def collate_batch(items: list[dict]) -> dict:
    return {
        "past": np.stack([item["past"] for item in items]).astype(np.float32),
        "future": np.stack([item["future"] for item in items]).astype(np.float32),
        "origin": np.stack([item["origin"] for item in items]).astype(np.float32),
        "meta": np.array([item["meta"] for item in items], dtype=items[0]["meta"].dtype),
    }


class NumpyDataLoader:
    def __init__(
        self,
        dataset: TrajectoryNPZDataset,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 42,
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __len__(self) -> int:
        return int(np.ceil(len(self.dataset) / self.batch_size))

    def __iter__(self) -> Iterator[dict]:
        indices = np.arange(len(self.dataset))
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            if len(self.dataset.chunk_paths) > 1:
                chunk_indices = []
                chunk_order = rng.permutation(len(self.dataset.chunk_paths))
                chunk_start = np.concatenate(([0], self.dataset.cumulative_lengths[:-1]))
                for chunk_index in chunk_order:
                    local_indices = np.arange(chunk_start[chunk_index], self.dataset.cumulative_lengths[chunk_index])
                    rng.shuffle(local_indices)
                    chunk_indices.append(local_indices)
                indices = np.concatenate(chunk_indices)
            else:
                rng.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]
            yield collate_batch([self.dataset[int(index)] for index in batch_indices])


def split_path(config: DataConfig, split: str) -> Path:
    single_file = config.split_dir / f"{config.prefix}_{split}.npz"
    chunks_dir = config.split_dir / f"{config.prefix}_{split}_chunks"
    return single_file if single_file.exists() else chunks_dir


def get_or_create_stats(config: DataConfig) -> dict[str, np.ndarray] | None:
    if not config.normalize:
        return None
    if config.stats_path.exists():
        return load_stats(config.stats_path)
    stats = compute_normalization_stats(split_path(config, "train"))
    save_stats(stats, config.stats_path)
    return stats


def build_datasets(config: DataConfig) -> dict[str, TrajectoryNPZDataset]:
    stats = get_or_create_stats(config)
    return {
        split: TrajectoryNPZDataset(
            split_path(config, split),
            relative_xy=config.relative_xy,
            normalize=config.normalize,
            stats=stats,
        )
        for split in ["train", "val", "test"]
    }


def build_dataloaders(config: DataConfig) -> dict[str, NumpyDataLoader]:
    datasets = build_datasets(config)
    return {
        "train": NumpyDataLoader(datasets["train"], config.batch_size, config.shuffle_train, config.seed),
        "val": NumpyDataLoader(datasets["val"], config.batch_size, False, config.seed),
        "test": NumpyDataLoader(datasets["test"], config.batch_size, False, config.seed),
    }
