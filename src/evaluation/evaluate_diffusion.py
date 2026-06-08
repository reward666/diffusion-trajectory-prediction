from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.datasets.datamodule import build_dataloaders
from src.datasets.graph_normalization import load_graph_stats
from src.datasets.normalization import deltas_to_positions, load_stats
from src.evaluation.metrics import MetricAccumulator, trajectory_metrics
from src.training.checkpoint import load_checkpoint
from src.training.train_diffusion import build_data_config, build_model, get_feature_names, load_yaml, select_device, set_seed
from src.visualization.trajectory_plot import plot_trajectory_prediction


def decode_future(values: np.ndarray, stats: dict[str, np.ndarray], representation: str) -> np.ndarray:
    if representation == "delta":
        deltas = values * stats["future_delta_std"] + stats["future_delta_mean"]
        return deltas_to_positions(deltas)
    return values * stats["future_std"] + stats["future_mean"]


def invert_past_xy_normalization(values: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    if "past_std" in stats:
        return values * stats["past_std"][:2] + stats["past_mean"][:2]
    return values * stats["ego_std"][:2] + stats["ego_mean"][:2]


def save_report(path: Path, report: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def evaluate(
    config_path: Path,
    checkpoint_path: Path,
    num_samples: int,
    max_trajectories: int | None,
    num_plots: int,
    report_path: Path,
    figures_dir: Path,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    data_config = build_data_config(config)
    set_seed(data_config.seed)
    device = select_device(config["training"]["device"])
    loaders = build_dataloaders(data_config)
    test_loader = loaders["test"]
    feature_names = get_feature_names(test_loader)
    model = build_model(config, feature_names).to(device)
    checkpoint = load_checkpoint(checkpoint_path, model, map_location=device)
    model.eval()
    stats = load_graph_stats(data_config.stats_path) if data_config.dataset_type == "graph" else load_stats(data_config.stats_path)

    accumulator = MetricAccumulator()
    plotted = 0
    processed = 0
    for batch in tqdm(test_loader, desc="evaluate"):
        if max_trajectories is not None:
            remaining = max_trajectories - processed
            if remaining <= 0:
                break
            if len(batch["past"]) > remaining:
                batch = {key: value[:remaining] for key, value in batch.items()}

        if "past" in batch:
            condition_input = torch.from_numpy(batch["past"]).to(device=device, dtype=torch.float32)
            past_for_plot = batch["past"]
        else:
            condition_input = {
                key: torch.from_numpy(batch[key]).to(device=device, dtype=torch.float32)
                for key in ["ego_past", "neighbor_past", "edge_attr", "neighbor_mask"]
            }
            past_for_plot = batch["ego_past"]
        normalized_predictions = model.sample(condition_input, num_samples=num_samples).cpu().numpy()
        normalized_future = batch["future"]
        relative_predictions = decode_future(normalized_predictions, stats, data_config.future_representation)
        relative_future = decode_future(normalized_future, stats, data_config.future_representation)
        predictions_xy = relative_predictions + batch["origin"][:, None, None, :]
        future_xy = relative_future + batch["origin"][:, None, :]
        accumulator.update(trajectory_metrics(predictions_xy, future_xy))

        for index in range(len(past_for_plot)):
            if plotted >= num_plots:
                break
            relative_past_xy = invert_past_xy_normalization(past_for_plot[index, :, :2], stats)
            past_xy = relative_past_xy + batch["origin"][index]
            meta = batch["meta"][index]
            plot_trajectory_prediction(
                past_xy=past_xy,
                future_xy=future_xy[index],
                predictions_xy=predictions_xy[index],
                output_path=figures_dir / f"trajectory_{plotted:03d}.png",
                title=f"{meta['scene_id']} track={meta['track_id']} frame={meta['obs_end_frame']}",
            )
            plotted += 1
        processed += len(past_for_plot)

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "metrics": accumulator.compute(num_samples),
        "figures_dir": str(figures_dir),
    }
    save_report(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate and visualize diffusion trajectory predictions.")
    parser.add_argument("--config", type=Path, default=Path("configs/ngsim_leader_clean_diffusion.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints/diffusion_best.pt"))
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--num-plots", type=int, default=12)
    parser.add_argument("--report-path", type=Path, default=Path("outputs/reports/diffusion_test_metrics.json"))
    parser.add_argument("--figures-dir", type=Path, default=Path("outputs/figures/diffusion_test"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        max_trajectories=args.max_trajectories,
        num_plots=args.num_plots,
        report_path=args.report_path,
        figures_dir=args.figures_dir,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
