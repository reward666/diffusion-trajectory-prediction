from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.datasets.datamodule import build_dataloaders
from src.datasets.normalization import load_stats
from src.evaluation.evaluate_diffusion import decode_future, invert_past_xy_normalization
from src.evaluation.metrics import MetricAccumulator, trajectory_metrics
from src.inference.trajectory_scorer import RuleBasedScorerConfig, score_trajectory_candidates, select_best_candidates
from src.training.checkpoint import load_checkpoint
from src.training.train_diffusion import build_data_config, build_model, get_feature_names, load_yaml, select_device, set_seed


def invert_past_normalization(values: np.ndarray, stats: dict[str, np.ndarray]) -> np.ndarray:
    return values * stats["past_std"] + stats["past_mean"]


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
    report_path: Path,
    scorer_config: RuleBasedScorerConfig,
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
    stats = load_stats(data_config.stats_path)

    random_accumulator = MetricAccumulator()
    scored_accumulator = MetricAccumulator()
    oracle_accumulator = MetricAccumulator()
    penalty_sums: dict[str, float] = {}
    processed = 0

    for batch in tqdm(test_loader, desc="score candidates"):
        if max_trajectories is not None:
            remaining = max_trajectories - processed
            if remaining <= 0:
                break
            if len(batch["past"]) > remaining:
                batch = {key: value[:remaining] for key, value in batch.items()}

        past = torch.from_numpy(batch["past"]).to(device=device, dtype=torch.float32)
        normalized_predictions = model.sample(past, num_samples=num_samples).cpu().numpy()
        relative_predictions = decode_future(normalized_predictions, stats, data_config.future_representation)
        relative_future = decode_future(batch["future"], stats, data_config.future_representation)
        predictions_xy = relative_predictions + batch["origin"][:, None, None, :]
        future_xy = relative_future + batch["origin"][:, None, :]
        past_features = invert_past_normalization(batch["past"], stats)
        relative_past_xy = invert_past_xy_normalization(batch["past"][:, :, :2], stats)
        past_xy = relative_past_xy + batch["origin"][:, None, :]

        score_parts = score_trajectory_candidates(predictions_xy, past_xy, past_features, feature_names, scorer_config)
        selected, _ = select_best_candidates(predictions_xy, score_parts["total"])
        random_accumulator.update(trajectory_metrics(predictions_xy[:, :1], future_xy))
        scored_accumulator.update(trajectory_metrics(selected[:, None], future_xy))
        oracle_accumulator.update(trajectory_metrics(predictions_xy, future_xy))
        for name, values in score_parts.items():
            penalty_sums[name] = penalty_sums.get(name, 0.0) + float(values.mean(axis=1).sum())
        processed += len(batch["past"])

    report = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "scorer": vars(scorer_config),
        "metrics": {
            "random_top1": random_accumulator.compute(num_samples=1),
            "rule_scored_top1": scored_accumulator.compute(num_samples=1),
            f"oracle_min_at_{num_samples}": oracle_accumulator.compute(num_samples=num_samples),
        },
        "mean_candidate_penalties": {name: value / processed for name, value in penalty_sums.items()},
    }
    save_report(report_path, report)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate rule-scored diffusion trajectory candidates.")
    parser.add_argument("--config", type=Path, default=Path("configs/ngsim_diffusion.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints_leader/diffusion_best.pt"))
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--max-trajectories", type=int)
    parser.add_argument("--report-path", type=Path, default=Path("outputs/reports/scored_diffusion_test_metrics.json"))
    parser.add_argument("--fps", type=float, default=10.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = evaluate(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        num_samples=args.num_samples,
        max_trajectories=args.max_trajectories,
        report_path=args.report_path,
        scorer_config=RuleBasedScorerConfig(fps=args.fps),
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

