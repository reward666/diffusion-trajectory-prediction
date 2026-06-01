from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.optim import AdamW
from tqdm import tqdm

from src.datasets.datamodule import DataConfig, build_dataloaders
from src.models.diffusion.trajectory_diffusion import TrajectoryDiffusion
from src.training.checkpoint import load_checkpoint, save_checkpoint


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(configured_device: str) -> torch.device:
    if configured_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(configured_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device


def build_data_config(config: dict[str, Any]) -> DataConfig:
    data = config["data"]
    return DataConfig(
        split_dir=Path(data["split_dir"]),
        prefix=data["prefix"],
        batch_size=int(data["batch_size"]),
        shuffle_train=bool(data["shuffle_train"]),
        seed=int(data["seed"]),
        relative_xy=bool(data["relative_xy"]),
        normalize=bool(data["normalize"]),
        stats_path=Path(data["stats_path"]),
    )


def get_feature_names(loader) -> list[str]:
    return [str(name) for name in loader.dataset.feature_names.tolist()]


def build_model(config: dict[str, Any], feature_names: list[str]) -> TrajectoryDiffusion:
    model = config["model"]
    return TrajectoryDiffusion(
        feature_names=feature_names,
        pred_len=int(model["pred_len"]),
        future_dim=int(model["future_dim"]),
        condition_dim=int(model["condition_dim"]),
        time_dim=int(model["time_dim"]),
        denoiser_hidden_dim=int(model["denoiser_hidden_dim"]),
        denoiser_num_layers=int(model["denoiser_num_layers"]),
        num_train_timesteps=int(model["num_train_timesteps"]),
    )


def batch_to_torch(batch: dict, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    past = torch.from_numpy(batch["past"]).to(device=device, dtype=torch.float32)
    future = torch.from_numpy(batch["future"]).to(device=device, dtype=torch.float32)
    return past, future


def train_one_epoch(model, loader, optimizer, device: torch.device) -> float:
    model.train()
    losses = []
    for batch in tqdm(loader, desc="train", leave=False):
        past, future = batch_to_torch(batch, device)
        optimizer.zero_grad(set_to_none=True)
        loss = model.training_loss(past, future)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


@torch.no_grad()
def validate(model, loader, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in tqdm(loader, desc="val", leave=False):
        past, future = batch_to_torch(batch, device)
        loss = model.training_loss(past, future)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


def save_log(log_path: Path, log: list[dict[str, Any]]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)


def train(config: dict[str, Any]) -> None:
    data_config = build_data_config(config)
    set_seed(data_config.seed)
    device = select_device(config["training"]["device"])
    loaders = build_dataloaders(data_config)
    feature_names = get_feature_names(loaders["train"])
    model = build_model(config, feature_names).to(device)

    training = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    checkpoint_dir = Path(training["checkpoint_dir"])
    log_path = Path(training["log_path"])
    best_val_loss = float("inf")
    start_epoch = 1
    log: list[dict[str, Any]] = []

    resume_from = training.get("resume_from")
    if resume_from:
        checkpoint = load_checkpoint(resume_from, model, optimizer, map_location=device)
        start_epoch = int(checkpoint["epoch"]) + 1
        best_val_loss = float(checkpoint["best_val_loss"])

    print(f"device={device}")
    print(f"train_samples={len(loaders['train'].dataset)} val_samples={len(loaders['val'].dataset)}")
    print(f"feature_names={feature_names}")

    for epoch in range(start_epoch, int(training["epochs"]) + 1):
        train_loss = train_one_epoch(model, loaders["train"], optimizer, device)
        val_loss = validate(model, loaders["val"], device)
        row = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss}
        log.append(row)
        save_log(log_path, log)

        save_checkpoint(
            checkpoint_dir / "diffusion_last.pt",
            model,
            optimizer,
            epoch,
            min(best_val_loss, val_loss),
            config,
            feature_names,
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_checkpoint(
                checkpoint_dir / "diffusion_best.pt",
                model,
                optimizer,
                epoch,
                best_val_loss,
                config,
                feature_names,
            )
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the NGSIM trajectory diffusion model.")
    parser.add_argument("--config", type=Path, default=Path("configs/ngsim_diffusion.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train(load_yaml(args.config))


if __name__ == "__main__":
    main()

