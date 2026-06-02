from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference.scorer_candidates import generate_candidate_cache


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate fixed diffusion candidates for learned scorer training.")
    parser.add_argument("--config", type=Path, default=Path("configs/ngsim_diffusion.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=Path("outputs/checkpoints_leader/diffusion_best.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/scorer_candidates"))
    parser.add_argument("--splits", nargs="+", choices=["train", "val", "test"], default=["train", "val", "test"])
    parser.add_argument("--num-samples", type=int, default=6)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--max-trajectories", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = generate_candidate_cache(
        config_path=args.config,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        splits=args.splits,
        num_samples=args.num_samples,
        chunk_size=args.chunk_size,
        max_trajectories=args.max_trajectories,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

