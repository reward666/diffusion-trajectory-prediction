from __future__ import annotations

from pathlib import Path

import numpy as np

from src.datasets.normalization import positions_to_deltas


def iter_graph_npz_arrays(npz_path: str | Path):
    path = Path(npz_path)
    chunk_paths = [path] if path.is_file() else sorted(path.glob("chunk_*.npz"))
    if not chunk_paths:
        raise FileNotFoundError(f"No graph NPZ file or chunks found at {path}.")
    for chunk_path in chunk_paths:
        with np.load(chunk_path, allow_pickle=False) as data:
            yield (
                data["ego_past"].astype(np.float32),
                data["neighbor_past"].astype(np.float32),
                data["edge_attr"].astype(np.float32),
                data["future"].astype(np.float32),
            )


def compute_graph_normalization_stats(npz_path: str | Path, eps: float = 1e-6) -> dict[str, np.ndarray]:
    ego_sum = ego_sq_sum = neighbor_sum = neighbor_sq_sum = edge_sum = edge_sq_sum = None
    future_sum = future_sq_sum = delta_sum = delta_sq_sum = None
    ego_count = neighbor_count = edge_count = future_count = delta_count = 0

    for ego, neighbor, edge, future in iter_graph_npz_arrays(npz_path):
        origin = ego[:, -1:, :2].copy()
        ego_rel = ego.copy()
        neighbor_rel = neighbor.copy()
        future_rel = future.copy()
        ego_rel[:, :, :2] -= origin
        neighbor_rel[:, :, :, :2] -= origin[:, None, :, :]
        future_rel[:, :, :2] -= origin
        future_delta = positions_to_deltas(future_rel[:, :, :2])

        arrays = [
            ("ego", ego_rel, (0, 1)),
            ("neighbor", neighbor_rel, (0, 1, 2)),
            ("edge", edge, (0, 1)),
            ("future", future_rel, (0, 1)),
            ("delta", future_delta, (0, 1)),
        ]
        for name, values, axes in arrays:
            chunk_sum = values.sum(axis=axes, dtype=np.float64)
            chunk_sq = np.square(values, dtype=np.float64).sum(axis=axes)
            count = int(np.prod([values.shape[axis] for axis in axes]))
            if name == "ego":
                ego_sum = chunk_sum if ego_sum is None else ego_sum + chunk_sum
                ego_sq_sum = chunk_sq if ego_sq_sum is None else ego_sq_sum + chunk_sq
                ego_count += count
            elif name == "neighbor":
                neighbor_sum = chunk_sum if neighbor_sum is None else neighbor_sum + chunk_sum
                neighbor_sq_sum = chunk_sq if neighbor_sq_sum is None else neighbor_sq_sum + chunk_sq
                neighbor_count += count
            elif name == "edge":
                edge_sum = chunk_sum if edge_sum is None else edge_sum + chunk_sum
                edge_sq_sum = chunk_sq if edge_sq_sum is None else edge_sq_sum + chunk_sq
                edge_count += count
            elif name == "future":
                future_sum = chunk_sum if future_sum is None else future_sum + chunk_sum
                future_sq_sum = chunk_sq if future_sq_sum is None else future_sq_sum + chunk_sq
                future_count += count
            else:
                delta_sum = chunk_sum if delta_sum is None else delta_sum + chunk_sum
                delta_sq_sum = chunk_sq if delta_sq_sum is None else delta_sq_sum + chunk_sq
                delta_count += count

    def mean_std(total, total_sq, count):
        mean = total / count
        std = np.sqrt(np.maximum(total_sq / count - np.square(mean), 0.0)) + eps
        return mean.astype(np.float32), std.astype(np.float32)

    ego_mean, ego_std = mean_std(ego_sum, ego_sq_sum, ego_count)
    neighbor_mean, neighbor_std = mean_std(neighbor_sum, neighbor_sq_sum, neighbor_count)
    edge_mean, edge_std = mean_std(edge_sum, edge_sq_sum, edge_count)
    future_mean, future_std = mean_std(future_sum, future_sq_sum, future_count)
    delta_mean, delta_std = mean_std(delta_sum, delta_sq_sum, delta_count)
    return {
        "ego_mean": ego_mean,
        "ego_std": ego_std,
        "neighbor_mean": neighbor_mean,
        "neighbor_std": neighbor_std,
        "edge_mean": edge_mean,
        "edge_std": edge_std,
        "future_mean": future_mean,
        "future_std": future_std,
        "future_delta_mean": delta_mean,
        "future_delta_std": delta_std,
    }


def save_graph_stats(stats: dict[str, np.ndarray], output_path: str | Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **stats)


def load_graph_stats(stats_path: str | Path) -> dict[str, np.ndarray]:
    data = np.load(Path(stats_path), allow_pickle=False)
    return {key: data[key].astype(np.float32) for key in data.files}
