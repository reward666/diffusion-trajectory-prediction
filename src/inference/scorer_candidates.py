from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from src.datasets.datamodule import build_dataloaders
from src.datasets.normalization import load_stats
from src.evaluation.evaluate_diffusion import decode_future
from src.evaluation.metrics import displacement_errors
from src.training.checkpoint import load_checkpoint
from src.training.train_diffusion import build_data_config, build_model, get_feature_names, load_yaml, select_device, set_seed


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)


@torch.no_grad()
def generate_candidate_cache(
    config_path: Path,
    checkpoint_path: Path,
    output_dir: Path,
    splits: list[str],
    num_samples: int,
    chunk_size: int,
    max_trajectories: int | None,
) -> dict[str, Any]:
    config = load_yaml(config_path)
    data_config = build_data_config(config)
    set_seed(data_config.seed)
    device = select_device(config["training"]["device"])
    loaders = build_dataloaders(data_config)
    feature_names = get_feature_names(loaders["train"])
    diffusion = build_model(config, feature_names).to(device)
    checkpoint = load_checkpoint(checkpoint_path, diffusion, map_location=device)
    diffusion.eval()
    stats = load_stats(data_config.stats_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any] = {
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": int(checkpoint["epoch"]),
        "num_candidates": num_samples,
        "context_dim": int(len(feature_names)),
        "pred_len": int(config["model"]["pred_len"]),
        "future_dim": int(config["model"]["future_dim"]),
        "splits": {},
    }

    for split in splits:
        split_dir = output_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        if any(split_dir.glob("chunk_*.npz")):
            raise FileExistsError(
                f"{split_dir} already contains candidate chunks. "
                "Move or remove that directory before generating a new cache."
            )
        context_buffer: list[np.ndarray] = []
        candidate_buffer: list[np.ndarray] = []
        ade_buffer: list[np.ndarray] = []
        fde_buffer: list[np.ndarray] = []
        chunk_index = 0
        processed = 0

        for batch in tqdm(loaders[split], desc=f"generate {split}"):
            if max_trajectories is not None:
                remaining = max_trajectories - processed
                if remaining <= 0:
                    break
                if len(batch["past"]) > remaining:
                    batch = {key: value[:remaining] for key, value in batch.items()}

            past = torch.from_numpy(batch["past"]).to(device=device, dtype=torch.float32)
            normalized_predictions = diffusion.sample(past, num_samples=num_samples).cpu().numpy()
            relative_predictions = decode_future(normalized_predictions, stats, data_config.future_representation)
            relative_future = decode_future(batch["future"], stats, data_config.future_representation)
            displacement = displacement_errors(relative_predictions, relative_future)
            target_ade = displacement.mean(axis=-1)
            target_fde = displacement[:, :, -1]

            context_buffer.extend(batch["past"][:, -1, :].astype(np.float32))
            candidate_buffer.extend(relative_predictions.astype(np.float32))
            ade_buffer.extend(target_ade.astype(np.float32))
            fde_buffer.extend(target_fde.astype(np.float32))
            processed += len(batch["past"])

            while len(context_buffer) >= chunk_size:
                chunk_index = _write_chunk(split_dir, chunk_index, context_buffer, candidate_buffer, ade_buffer, fde_buffer, chunk_size)

        if context_buffer:
            chunk_index = _write_chunk(split_dir, chunk_index, context_buffer, candidate_buffer, ade_buffer, fde_buffer, len(context_buffer))

        manifest["splits"][split] = {
            "num_trajectories": processed,
            "num_chunks": chunk_index,
        }

    save_manifest(output_dir / "manifest.json", manifest)
    return manifest


def _write_chunk(
    split_dir: Path,
    chunk_index: int,
    context_buffer: list[np.ndarray],
    candidate_buffer: list[np.ndarray],
    ade_buffer: list[np.ndarray],
    fde_buffer: list[np.ndarray],
    count: int,
) -> int:
    np.savez_compressed(
        split_dir / f"chunk_{chunk_index:05d}.npz",
        context=np.stack(context_buffer[:count]),
        candidates=np.stack(candidate_buffer[:count]),
        target_ade=np.stack(ade_buffer[:count]),
        target_fde=np.stack(fde_buffer[:count]),
    )
    del context_buffer[:count]
    del candidate_buffer[:count]
    del ade_buffer[:count]
    del fde_buffer[:count]
    return chunk_index + 1


class CandidateCacheDataset:
    def __init__(self, split_dir: str | Path):
        self.chunk_paths = sorted(Path(split_dir).glob("chunk_*.npz"))
        if not self.chunk_paths:
            raise FileNotFoundError(f"No candidate chunks found under {split_dir}.")
        self.chunk_lengths = []
        self._cached_chunk_index = -1
        self._cached_data = None
        for path in self.chunk_paths:
            with np.load(path, allow_pickle=False) as data:
                self.chunk_lengths.append(int(data["context"].shape[0]))
        self.cumulative_lengths = np.cumsum(self.chunk_lengths)

    def __len__(self) -> int:
        return int(self.cumulative_lengths[-1])

    def __getitem__(self, index: int) -> dict[str, np.ndarray]:
        chunk_index = int(np.searchsorted(self.cumulative_lengths, index, side="right"))
        chunk_start = 0 if chunk_index == 0 else int(self.cumulative_lengths[chunk_index - 1])
        if chunk_index != self._cached_chunk_index:
            with np.load(self.chunk_paths[chunk_index], allow_pickle=False) as data:
                self._cached_data = {key: data[key] for key in data.files}
            self._cached_chunk_index = chunk_index
        local_index = index - chunk_start
        return {key: value[local_index] for key, value in self._cached_data.items()}
