from __future__ import annotations

import numpy as np


def displacement_errors(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    if predictions.ndim != 4:
        raise ValueError(f"Expected predictions shape [batch, k, pred_len, 2], got {predictions.shape}.")
    if target.ndim != 3:
        raise ValueError(f"Expected target shape [batch, pred_len, 2], got {target.shape}.")
    return np.linalg.norm(predictions - target[:, None, :, :], axis=-1)


def trajectory_metrics(predictions: np.ndarray, target: np.ndarray) -> dict[str, np.ndarray]:
    errors = displacement_errors(predictions, target)
    ade_by_candidate = errors.mean(axis=-1)
    fde_by_candidate = errors[:, :, -1]
    return {
        "ade": ade_by_candidate[:, 0],
        "fde": fde_by_candidate[:, 0],
        "min_ade": ade_by_candidate.min(axis=1),
        "min_fde": fde_by_candidate.min(axis=1),
    }


class MetricAccumulator:
    def __init__(self):
        self.sums = {"ade": 0.0, "fde": 0.0, "min_ade": 0.0, "min_fde": 0.0}
        self.count = 0

    def update(self, metrics: dict[str, np.ndarray]) -> None:
        batch_size = len(metrics["ade"])
        self.count += batch_size
        for name in self.sums:
            self.sums[name] += float(metrics[name].sum())

    def compute(self, num_samples: int) -> dict[str, float | int]:
        if self.count == 0:
            raise ValueError("No samples were accumulated.")
        return {
            "num_trajectories": self.count,
            "num_candidates": num_samples,
            "ade_m": self.sums["ade"] / self.count,
            "fde_m": self.sums["fde"] / self.count,
            f"min_ade_at_{num_samples}_m": self.sums["min_ade"] / self.count,
            f"min_fde_at_{num_samples}_m": self.sums["min_fde"] / self.count,
        }

