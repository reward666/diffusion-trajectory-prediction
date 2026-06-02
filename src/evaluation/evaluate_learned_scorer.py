from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.evaluation.metrics import MetricAccumulator
from src.inference.learned_scorer import LearnedTrajectoryScorer
from src.inference.scorer_candidates import CandidateCacheDataset
from src.training.train_learned_scorer import iter_batches
from src.training.train_diffusion import select_device


@torch.no_grad()
def evaluate(args: argparse.Namespace) -> dict:
    device = select_device(args.device)
    dataset = CandidateCacheDataset(args.candidates_dir / "test")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model = LearnedTrajectoryScorer(
        checkpoint["context_dim"],
        checkpoint["pred_len"],
        checkpoint["future_dim"],
        checkpoint["hidden_dim"],
        checkpoint["dropout"],
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    random_accumulator = MetricAccumulator()
    learned_accumulator = MetricAccumulator()
    oracle_accumulator = MetricAccumulator()
    for batch in tqdm(iter_batches(dataset, args.batch_size, False, 42), desc="evaluate learned scorer"):
        context = torch.from_numpy(batch["context"]).to(device=device, dtype=torch.float32)
        candidates = torch.from_numpy(batch["candidates"]).to(device=device, dtype=torch.float32)
        logits = model(context, candidates).cpu().numpy()
        best_indices = logits.argmax(axis=1)
        random_metrics = _metrics_from_errors(batch["target_ade"][:, :1], batch["target_fde"][:, :1])
        selected_ade = batch["target_ade"][np.arange(len(best_indices)), best_indices]
        selected_fde = batch["target_fde"][np.arange(len(best_indices)), best_indices]
        learned_metrics = _metrics_from_errors(selected_ade[:, None], selected_fde[:, None])
        oracle_metrics = _metrics_from_errors(batch["target_ade"], batch["target_fde"])
        random_accumulator.update(random_metrics)
        learned_accumulator.update(learned_metrics)
        oracle_accumulator.update(oracle_metrics)

    num_candidates = int(batch["candidates"].shape[1])
    report = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "best_val_loss": float(checkpoint["best_val_loss"]),
        "metrics": {
            "random_top1": random_accumulator.compute(1),
            "learned_top1": learned_accumulator.compute(1),
            f"oracle_min_at_{num_candidates}": oracle_accumulator.compute(num_candidates),
        },
    }
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    with args.report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    return report


def _metrics_from_errors(ade_errors: np.ndarray, fde_errors: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "ade": ade_errors[:, 0],
        "fde": fde_errors[:, 0],
        "min_ade": ade_errors.min(axis=1),
        "min_fde": fde_errors.min(axis=1),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a learned candidate scorer.")
    parser.add_argument("--candidates-dir", type=Path, default=Path("outputs/scorer_candidates"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints_scorer/scorer_best.pt"))
    parser.add_argument("--report-path", type=Path, default=Path("outputs/reports/learned_scorer_metrics.json"))
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    report = evaluate(parse_args())
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
