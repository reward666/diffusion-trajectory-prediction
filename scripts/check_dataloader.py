from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.datamodule import DataConfig, build_dataloaders


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check trajectory NPZ dataloaders.")
    parser.add_argument("--split-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--prefix", type=str, default="ngsim")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--no-normalize", action="store_true")
    parser.add_argument("--absolute-xy", action="store_true")
    parser.add_argument("--future-representation", choices=["position", "delta"], default="position")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = DataConfig(
        split_dir=args.split_dir,
        prefix=args.prefix,
        batch_size=args.batch_size,
        normalize=not args.no_normalize,
        relative_xy=not args.absolute_xy,
        future_representation=args.future_representation,
        stats_path=args.split_dir / f"{args.prefix}_stats.npz",
    )
    loaders = build_dataloaders(config)
    summary = {}
    for split, loader in loaders.items():
        batch = next(iter(loader))
        summary[split] = {
            "num_batches": len(loader),
            "batch_past_shape": list(batch["past"].shape),
            "batch_future_shape": list(batch["future"].shape),
            "batch_origin_shape": list(batch["origin"].shape),
        }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
