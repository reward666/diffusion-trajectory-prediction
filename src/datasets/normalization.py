from __future__ import annotations

from pathlib import Path

import numpy as np


def iter_npz_arrays(npz_path: str | Path):
    path = Path(npz_path)
    chunk_paths = [path] if path.is_file() else sorted(path.glob("chunk_*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(f"No trajectory NPZ file or chunks found at {path}.")
    for chunk_path in chunk_paths:
        data = np.load(chunk_path, allow_pickle=False)
        yield data["past"].astype(np.float32), data["future"].astype(np.float32)


def to_relative_xy(past: np.ndarray, future: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    past_rel = past.astype(np.float32, copy=True)
    future_rel = future.astype(np.float32, copy=True)
    origins = past_rel[:, -1:, :2].copy()
    past_rel[:, :, :2] -= origins
    future_rel[:, :, :2] -= origins
    return past_rel, future_rel


def positions_to_deltas(future_xy: np.ndarray) -> np.ndarray:
    start = np.zeros_like(future_xy[:, :1, :])
    return np.diff(np.concatenate([start, future_xy], axis=1), axis=1)


def deltas_to_positions(future_delta: np.ndarray) -> np.ndarray:
    return np.cumsum(future_delta, axis=-2)


def compute_normalization_stats(npz_path: str | Path, eps: float = 1e-6) -> dict[str, np.ndarray]:
    past_sum = past_sq_sum = future_sum = future_sq_sum = None
    delta_sum = delta_sq_sum = None
    past_count = future_count = delta_count = 0
    for past, future in iter_npz_arrays(npz_path):
        past_rel, future_rel = to_relative_xy(past, future)
        future_delta = positions_to_deltas(future_rel)
        chunk_past_sum = past_rel.sum(axis=(0, 1), dtype=np.float64)
        chunk_future_sum = future_rel.sum(axis=(0, 1), dtype=np.float64)
        chunk_delta_sum = future_delta.sum(axis=(0, 1), dtype=np.float64)
        chunk_past_sq_sum = np.square(past_rel, dtype=np.float64).sum(axis=(0, 1))
        chunk_future_sq_sum = np.square(future_rel, dtype=np.float64).sum(axis=(0, 1))
        chunk_delta_sq_sum = np.square(future_delta, dtype=np.float64).sum(axis=(0, 1))
        past_sum = chunk_past_sum if past_sum is None else past_sum + chunk_past_sum
        future_sum = chunk_future_sum if future_sum is None else future_sum + chunk_future_sum
        delta_sum = chunk_delta_sum if delta_sum is None else delta_sum + chunk_delta_sum
        past_sq_sum = chunk_past_sq_sum if past_sq_sum is None else past_sq_sum + chunk_past_sq_sum
        future_sq_sum = chunk_future_sq_sum if future_sq_sum is None else future_sq_sum + chunk_future_sq_sum
        delta_sq_sum = chunk_delta_sq_sum if delta_sq_sum is None else delta_sq_sum + chunk_delta_sq_sum
        past_count += past_rel.shape[0] * past_rel.shape[1]
        future_count += future_rel.shape[0] * future_rel.shape[1]
        delta_count += future_delta.shape[0] * future_delta.shape[1]

    past_mean = past_sum / past_count
    future_mean = future_sum / future_count
    delta_mean = delta_sum / delta_count
    stats = {
        "past_mean": past_mean.astype(np.float32),
        "past_std": np.sqrt(np.maximum(past_sq_sum / past_count - np.square(past_mean), 0.0)).astype(np.float32) + eps,
        "future_mean": future_mean.astype(np.float32),
        "future_std": np.sqrt(np.maximum(future_sq_sum / future_count - np.square(future_mean), 0.0)).astype(np.float32) + eps,
        "future_delta_mean": delta_mean.astype(np.float32),
        "future_delta_std": np.sqrt(np.maximum(delta_sq_sum / delta_count - np.square(delta_mean), 0.0)).astype(np.float32) + eps,
    }
    return stats


def save_stats(stats: dict[str, np.ndarray], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **stats)


def load_stats(stats_path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(stats_path), allow_pickle=False)
    return {key: data[key].astype(np.float32) for key in data.files}
