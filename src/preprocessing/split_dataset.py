from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SplitConfig:
    input_npz: Path
    output_dir: Path
    prefix: str = "ngsim"
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    seed: int = 42


def validate_ratios(config: SplitConfig) -> None:
    total = config.train_ratio + config.val_ratio + config.test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}.")
    if min(config.train_ratio, config.val_ratio, config.test_ratio) < 0:
        raise ValueError("Split ratios must be non-negative.")


def find_sample_chunks(input_npz: Path) -> list[Path]:
    if input_npz.is_file():
        return [input_npz]
    if input_npz.is_dir():
        chunks = sorted(input_npz.glob("chunk_*.npz"))
        if chunks:
            return chunks
    raise FileNotFoundError(f"No sample NPZ file or chunks found at {input_npz}")


def load_samples(input_npz: Path) -> dict[str, np.ndarray]:
    data = np.load(input_npz, allow_pickle=False)
    required = ["past", "future", "meta", "feature_names"]
    missing = [key for key in required if key not in data.files]
    if missing:
        raise ValueError(f"{input_npz} is missing arrays: {missing}")
    return {key: data[key] for key in data.files}


def get_group_keys(meta: np.ndarray) -> np.ndarray:
    required = {"scene_id", "source_file", "track_id"}
    if not required.issubset(meta.dtype.names or []):
        raise ValueError(f"meta dtype must contain {sorted(required)}.")

    keys = np.array(
        [
            f"{row['scene_id']}|{row['source_file']}|{row['track_id']}"
            for row in meta
        ]
    )
    return keys


def assign_groups(keys: np.ndarray, config: SplitConfig) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(config.seed)
    unique_groups = np.unique(keys)
    rng.shuffle(unique_groups)

    n_groups = len(unique_groups)
    n_train = int(round(n_groups * config.train_ratio))
    n_val = int(round(n_groups * config.val_ratio))
    n_train = min(n_train, n_groups)
    n_val = min(n_val, n_groups - n_train)

    train_groups = unique_groups[:n_train]
    val_groups = unique_groups[n_train : n_train + n_val]
    test_groups = unique_groups[n_train + n_val :]

    return {
        "train": train_groups,
        "val": val_groups,
        "test": test_groups,
    }


def subset_arrays(data: dict[str, np.ndarray], mask: np.ndarray) -> dict[str, np.ndarray]:
    subset = {}
    n_samples = data["past"].shape[0]
    for key, value in data.items():
        if value.shape[:1] == (n_samples,):
            subset[key] = value[mask]
        else:
            subset[key] = value
    return subset


def save_split_chunks(
    chunk_paths: list[Path],
    groups_by_split: dict[str, np.ndarray],
    config: SplitConfig,
) -> dict:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    split_dirs = {}
    split_info = {}
    for split_name, split_groups in groups_by_split.items():
        split_dir = config.output_dir / f"{config.prefix}_{split_name}_chunks"
        if split_dir.exists() and any(split_dir.glob("chunk_*.npz")):
            raise FileExistsError(f"{split_dir} already contains chunks. Move or remove it before splitting again.")
        split_dir.mkdir(parents=True, exist_ok=True)
        split_dirs[split_name] = split_dir
        split_info[split_name] = {
            "num_groups": int(len(split_groups)),
            "num_samples": 0,
            "num_chunks": 0,
            "scenes": {},
        }

    num_samples = 0
    for chunk_index, chunk_path in enumerate(chunk_paths):
        print(f"[{chunk_index + 1}/{len(chunk_paths)}] splitting {chunk_path.name}", flush=True)
        data = load_samples(chunk_path)
        keys = get_group_keys(data["meta"])
        num_samples += int(data["past"].shape[0])
        for split_name, split_groups in groups_by_split.items():
            mask = np.isin(keys, split_groups)
            if not mask.any():
                continue
            subset = subset_arrays(data, mask)
            output_index = split_info[split_name]["num_chunks"]
            np.savez_compressed(split_dirs[split_name] / f"chunk_{output_index:05d}.npz", **subset)
            split_info[split_name]["num_chunks"] += 1
            split_info[split_name]["num_samples"] += int(mask.sum())
            scenes, counts = np.unique(subset["meta"]["scene_id"], return_counts=True)
            for scene, count in zip(scenes, counts):
                scene = str(scene)
                split_info[split_name]["scenes"][scene] = split_info[split_name]["scenes"].get(scene, 0) + int(count)

    metadata = {
        "input_npz": str(config.input_npz),
        "output_dir": str(config.output_dir),
        "config": {
            **asdict(config),
            "input_npz": str(config.input_npz),
            "output_dir": str(config.output_dir),
        },
        "num_samples": int(num_samples),
        "num_groups": int(sum(len(groups) for groups in groups_by_split.values())),
        "splits": split_info,
    }
    with (config.output_dir / f"{config.prefix}_split_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def split_dataset(config: SplitConfig) -> dict:
    validate_ratios(config)
    chunk_paths = find_sample_chunks(config.input_npz)
    all_keys = []
    for chunk_path in chunk_paths:
        data = load_samples(chunk_path)
        all_keys.append(get_group_keys(data["meta"]))
    groups_by_split = assign_groups(np.concatenate(all_keys), config)
    return save_split_chunks(chunk_paths, groups_by_split, config)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split trajectory samples by scene/source/track group.")
    parser.add_argument("--input-npz", type=Path, default=Path("data/processed/ngsim/samples_chunks"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--prefix", type=str, default="ngsim")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = SplitConfig(
        input_npz=args.input_npz,
        output_dir=args.output_dir,
        prefix=args.prefix,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )
    metadata = split_dataset(config)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
