from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.optim import AdamW
from tqdm import tqdm

from src.inference.learned_scorer import LearnedTrajectoryScorer
from src.inference.scorer_candidates import CandidateCacheDataset
from src.training.train_diffusion import select_device, set_seed


def iter_batches(dataset: CandidateCacheDataset, batch_size: int, shuffle: bool, seed: int):
    indices = np.arange(len(dataset))
    if shuffle:
        rng = np.random.default_rng(seed)
        chunk_start = np.concatenate(([0], dataset.cumulative_lengths[:-1]))
        shuffled_chunks = []
        for chunk_index in rng.permutation(len(dataset.chunk_paths)):
            local_indices = np.arange(chunk_start[chunk_index], dataset.cumulative_lengths[chunk_index])
            rng.shuffle(local_indices)
            shuffled_chunks.append(local_indices)
        indices = np.concatenate(shuffled_chunks)
    for start in range(0, len(indices), batch_size):
        items = [dataset[int(index)] for index in indices[start : start + batch_size]]
        yield {
            "context": np.stack([item["context"] for item in items]),
            "candidates": np.stack([item["candidates"] for item in items]),
            "target_ade": np.stack([item["target_ade"] for item in items]),
            "target_fde": np.stack([item["target_fde"] for item in items]),
        }


def loss_fn(logits: torch.Tensor, target_errors: torch.Tensor, temperature: float) -> torch.Tensor:
    target_probabilities = torch.softmax(-target_errors / temperature, dim=1)
    return -(target_probabilities * torch.log_softmax(logits, dim=1)).sum(dim=1).mean()


def run_epoch(model, dataset, batch_size, device, temperature, optimizer=None, seed=42) -> float:
    is_training = optimizer is not None
    model.train(is_training)
    losses = []
    batches = iter_batches(dataset, batch_size, shuffle=is_training, seed=seed)
    for batch in tqdm(batches, desc="scorer train" if is_training else "scorer val", leave=False):
        context = torch.from_numpy(batch["context"]).to(device=device, dtype=torch.float32)
        candidates = torch.from_numpy(batch["candidates"]).to(device=device, dtype=torch.float32)
        target_errors = torch.from_numpy(batch["target_ade"]).to(device=device, dtype=torch.float32)
        with torch.set_grad_enabled(is_training):
            logits = model(context, candidates)
            loss = loss_fn(logits, target_errors, temperature)
            if is_training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def train(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = select_device(args.device)
    train_dataset = CandidateCacheDataset(args.candidates_dir / "train")
    val_dataset = CandidateCacheDataset(args.candidates_dir / "val")
    with np.load(train_dataset.chunk_paths[0], allow_pickle=False) as data:
        context_dim = int(data["context"].shape[-1])
        pred_len = int(data["candidates"].shape[-2])
        future_dim = int(data["candidates"].shape[-1])

    model = LearnedTrajectoryScorer(context_dim, pred_len, future_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_val_loss = float("inf")
    log = []

    for epoch in range(1, args.epochs + 1):
        train_loss = run_epoch(model, train_dataset, args.batch_size, device, args.temperature, optimizer, args.seed + epoch)
        with torch.no_grad():
            val_loss = run_epoch(model, val_dataset, args.batch_size, device, args.temperature)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        log.append(row)
        with (args.output_dir / "train_log.json").open("w", encoding="utf-8") as f:
            json.dump(log, f, indent=2)
        checkpoint = {
            "epoch": epoch,
            "best_val_loss": min(best_val_loss, val_loss),
            "model_state_dict": model.state_dict(),
            "context_dim": context_dim,
            "pred_len": pred_len,
            "future_dim": future_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
        }
        torch.save(checkpoint, args.output_dir / "scorer_last.pt")
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, args.output_dir / "scorer_best.pt")
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a learned diffusion-candidate scorer.")
    parser.add_argument("--candidates-dir", type=Path, default=Path("outputs/scorer_candidates"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/checkpoints_scorer"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto")
    return parser.parse_args()


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
