from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_trajectory_prediction(
    past_xy: np.ndarray,
    future_xy: np.ndarray,
    predictions_xy: np.ndarray,
    output_path: str | Path,
    title: str,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(past_xy[:, 0], past_xy[:, 1], color="#2563eb", linewidth=2.5, label="history")
    ax.plot(future_xy[:, 0], future_xy[:, 1], color="#16a34a", linewidth=2.5, label="ground truth")
    for index, prediction in enumerate(predictions_xy):
        ax.plot(
            prediction[:, 0],
            prediction[:, 1],
            color="#dc2626",
            linewidth=1.2,
            alpha=0.45,
            label="diffusion samples" if index == 0 else None,
        )

    ax.scatter(past_xy[-1, 0], past_xy[-1, 1], color="#111827", s=36, zorder=3, label="current position")
    ax.set_title(title)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.axis("equal")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

