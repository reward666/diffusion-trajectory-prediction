from __future__ import annotations

import numpy as np


def predict_constant_velocity(
    past: np.ndarray,
    pred_len: int,
    fps: float = 10.0,
    velocity_window: int = 5,
) -> np.ndarray:
    if past.ndim != 3:
        raise ValueError(f"Expected past shape [batch, obs_len, feature_dim], got {past.shape}.")
    if pred_len <= 0:
        raise ValueError("pred_len must be positive.")
    if velocity_window <= 0:
        raise ValueError("velocity_window must be positive.")

    window = min(velocity_window, past.shape[1])
    velocity = past[:, -window:, 2:4].mean(axis=1)
    seconds = np.arange(1, pred_len + 1, dtype=np.float32) / fps
    return past[:, -1:, :2] + velocity[:, None, :] * seconds[None, :, None]

