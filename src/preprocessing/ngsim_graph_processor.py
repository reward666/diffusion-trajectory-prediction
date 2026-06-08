from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from src.preprocessing.ngsim_processor import _clean_columns, _numeric_or_default, find_ngsim_files, read_ngsim_table
from src.preprocessing.ngsim_schema import NGSIM_COLUMNS


NODE_FEATURE_NAMES = ["x", "y", "vx", "vy", "speed", "acc"]
EDGE_FEATURE_NAMES = ["dx", "dy", "distance", "inverse_distance", "dvx", "dvy"]


@dataclass(frozen=True)
class NGSIMGraphProcessConfig:
    raw_dir: Path
    output_dir: Path
    location: str = "lankershim"
    obs_len: int = 30
    pred_len: int = 50
    stride: int = 20
    fps: float = 10.0
    radius_m: float = 30.0
    max_neighbors: int = 12
    chunk_size: int = 10000
    convert_feet_to_meters: bool = True

    @property
    def total_len(self) -> int:
        return self.obs_len + self.pred_len


def load_graph_tracks(path: Path, raw_dir: Path, config: NGSIMGraphProcessConfig) -> pd.DataFrame:
    df = _clean_columns(read_ngsim_table(path))
    cols = NGSIM_COLUMNS
    required = [cols.vehicle_id, cols.frame_id, cols.local_x, cols.local_y, cols.location]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required NGSIM columns: {missing}")

    scene_id = df[cols.location].astype(str).str.strip().str.lower()
    out = pd.DataFrame(
        {
            "scene_id": scene_id,
            "source_file": str(path.relative_to(raw_dir)),
            "frame": pd.to_numeric(df[cols.frame_id], errors="coerce"),
            "track_id": df[cols.vehicle_id].astype(str),
            "x": pd.to_numeric(df[cols.local_x], errors="coerce"),
            "y": pd.to_numeric(df[cols.local_y], errors="coerce"),
            "speed": _numeric_or_default(df, cols.speed),
            "acc": _numeric_or_default(df, cols.acc),
        }
    )
    out = out.dropna(subset=["frame", "x", "y"]).copy()
    out["frame"] = out["frame"].astype(int)
    out = out[out["scene_id"] == config.location.strip().lower()].copy()
    if out.empty:
        return out

    if config.convert_feet_to_meters:
        for col in ["x", "y", "speed", "acc"]:
            out[col] = out[col] * 0.3048

    if cols.global_time in df.columns:
        global_time = pd.to_numeric(df.loc[out.index, cols.global_time], errors="coerce")
        out["t"] = np.where(global_time.notna(), global_time / 1000.0, out["frame"] / config.fps)
    else:
        out["t"] = out["frame"] / config.fps

    out = out.sort_values(["scene_id", "source_file", "track_id", "frame"])
    grouped = out.groupby(["scene_id", "source_file", "track_id"], sort=False)
    dt = grouped["t"].diff()
    out["vx"] = grouped["x"].diff() / dt
    out["vy"] = grouped["y"].diff() / dt
    out[["vx", "vy"]] = out[["vx", "vy"]].replace([np.inf, -np.inf], np.nan)
    out[["vx", "vy"]] = grouped[["vx", "vy"]].transform(lambda s: s.bfill().ffill()).fillna(0.0)
    out["speed"] = out["speed"].fillna(np.hypot(out["vx"], out["vy"]))
    out["acc"] = out["acc"].fillna(0.0)
    return out[["scene_id", "source_file", "frame", "track_id", "t", *NODE_FEATURE_NAMES]]


def _continuous_track_windows(track: pd.DataFrame, config: NGSIMGraphProcessConfig):
    track = track.sort_values("frame").reset_index(drop=True)
    frames = track["frame"].to_numpy()
    if len(track) < config.total_len:
        return
    for start in range(0, len(track) - config.total_len + 1, config.stride):
        end = start + config.total_len
        if np.array_equal(frames[start:end], np.arange(frames[start], frames[start] + config.total_len)):
            yield track.iloc[start:end]


def _save_chunk(
    samples_dir: Path,
    chunk_index: int,
    ego_past: list[np.ndarray],
    neighbor_past: list[np.ndarray],
    edge_attr: list[np.ndarray],
    neighbor_mask: list[np.ndarray],
    future: list[np.ndarray],
    meta: list[tuple],
) -> int:
    if not ego_past:
        return chunk_index
    samples_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        samples_dir / f"chunk_{chunk_index:05d}.npz",
        ego_past=np.stack(ego_past),
        neighbor_past=np.stack(neighbor_past),
        edge_attr=np.stack(edge_attr),
        neighbor_mask=np.stack(neighbor_mask),
        future=np.stack(future),
        meta=np.array(
            meta,
            dtype=[
                ("scene_id", "U128"),
                ("source_file", "U256"),
                ("track_id", "U64"),
                ("obs_start_frame", "i8"),
                ("obs_end_frame", "i8"),
                ("pred_end_frame", "i8"),
            ],
        ),
        node_feature_names=np.array(NODE_FEATURE_NAMES),
        edge_feature_names=np.array(EDGE_FEATURE_NAMES),
    )
    return chunk_index + 1


def _neighbor_arrays(
    obs: pd.DataFrame,
    frame_lookup: dict[int, pd.DataFrame],
    track_lookup: dict[str, pd.DataFrame],
    config: NGSIMGraphProcessConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    current = obs.iloc[-1]
    frame_tracks = frame_lookup.get(int(current["frame"]))
    neighbor_past = np.zeros((config.max_neighbors, config.obs_len, len(NODE_FEATURE_NAMES)), dtype=np.float32)
    edge_attr = np.zeros((config.max_neighbors, len(EDGE_FEATURE_NAMES)), dtype=np.float32)
    neighbor_mask = np.zeros((config.max_neighbors,), dtype=np.float32)
    if frame_tracks is None or len(frame_tracks) <= 1:
        return neighbor_past, edge_attr, neighbor_mask

    candidates = frame_tracks[frame_tracks["track_id"] != current["track_id"]].copy()
    dx = candidates["x"].to_numpy() - float(current["x"])
    dy = candidates["y"].to_numpy() - float(current["y"])
    distance = np.hypot(dx, dy)
    valid = distance <= config.radius_m
    if not valid.any():
        return neighbor_past, edge_attr, neighbor_mask

    candidates = candidates.iloc[np.flatnonzero(valid)].copy()
    distance = distance[valid]
    order = np.argsort(distance)[: config.max_neighbors]
    obs_frames = obs["frame"].to_numpy()
    for slot, candidate_index in enumerate(order):
        candidate = candidates.iloc[int(candidate_index)]
        neighbor_track = track_lookup.get(str(candidate["track_id"]))
        if neighbor_track is None:
            continue
        neighbor_obs = neighbor_track[neighbor_track["frame"].isin(obs_frames)].sort_values("frame")
        if len(neighbor_obs) != config.obs_len or not np.array_equal(neighbor_obs["frame"].to_numpy(), obs_frames):
            continue
        rel_x = float(candidate["x"] - current["x"])
        rel_y = float(candidate["y"] - current["y"])
        dist = float(np.hypot(rel_x, rel_y))
        neighbor_past[slot] = neighbor_obs[NODE_FEATURE_NAMES].to_numpy(dtype=np.float32)
        edge_attr[slot] = np.array(
            [
                rel_x,
                rel_y,
                dist,
                1.0 / max(dist, 1.0),
                float(candidate["vx"] - current["vx"]),
                float(candidate["vy"] - current["vy"]),
            ],
            dtype=np.float32,
        )
        neighbor_mask[slot] = 1.0
    return neighbor_past, edge_attr, neighbor_mask


def process_ngsim_graph(config: NGSIMGraphProcessConfig) -> dict:
    files = find_ngsim_files(config.raw_dir)
    if not files:
        raise FileNotFoundError(f"No NGSIM CSV/TXT files found under {config.raw_dir}.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = config.output_dir / "samples_chunks"
    if samples_dir.exists() and any(samples_dir.glob("chunk_*.npz")):
        raise FileExistsError(f"{samples_dir} already contains chunks. Move or remove it before processing.")

    buffers = {
        "ego_past": [],
        "neighbor_past": [],
        "edge_attr": [],
        "neighbor_mask": [],
        "future": [],
        "meta": [],
    }
    num_rows = num_tracks = num_samples = chunk_index = 0
    skipped = []
    for file_index, path in enumerate(files, start=1):
        print(f"[{file_index}/{len(files)}] loading {path}", flush=True)
        try:
            tracks = load_graph_tracks(path, config.raw_dir, config)
        except ValueError as exc:
            skipped.append({"file": str(path), "reason": str(exc)})
            continue
        if tracks.empty:
            continue
        num_rows += len(tracks)
        num_tracks += int(tracks[["source_file", "track_id"]].drop_duplicates().shape[0])
        frame_lookup = {int(frame): frame_tracks for frame, frame_tracks in tracks.groupby("frame", sort=False)}
        track_lookup = {str(track_id): track for track_id, track in tracks.groupby("track_id", sort=False)}
        for track_id, track in track_lookup.items():
            for window in _continuous_track_windows(track, config):
                obs = window.iloc[: config.obs_len]
                pred = window.iloc[config.obs_len :]
                neighbors, edges, mask = _neighbor_arrays(obs, frame_lookup, track_lookup, config)
                buffers["ego_past"].append(obs[NODE_FEATURE_NAMES].to_numpy(dtype=np.float32))
                buffers["neighbor_past"].append(neighbors)
                buffers["edge_attr"].append(edges)
                buffers["neighbor_mask"].append(mask)
                buffers["future"].append(pred[["x", "y"]].to_numpy(dtype=np.float32))
                buffers["meta"].append(
                    (
                        str(obs["scene_id"].iloc[0]),
                        str(obs["source_file"].iloc[0]),
                        str(track_id),
                        int(obs["frame"].iloc[0]),
                        int(obs["frame"].iloc[-1]),
                        int(pred["frame"].iloc[-1]),
                    )
                )
                num_samples += 1
                if len(buffers["ego_past"]) >= config.chunk_size:
                    chunk_index = _save_chunk(samples_dir, chunk_index, **buffers)
                    print(f"  wrote chunk_{chunk_index - 1:05d}.npz ({num_samples} samples total)", flush=True)
                    for values in buffers.values():
                        values.clear()
        del tracks

    chunk_index = _save_chunk(samples_dir, chunk_index, **buffers)
    if num_samples == 0:
        raise ValueError("No graph trajectory samples were created.")

    metadata = {
        "dataset": "ngsim_graph",
        "raw_dir": str(config.raw_dir),
        "samples_dir": str(samples_dir),
        "num_files": len(files),
        "num_skipped_files": len(skipped),
        "skipped_files": skipped,
        "num_rows": int(num_rows),
        "num_tracks": int(num_tracks),
        "num_samples": int(num_samples),
        "num_chunks": int(chunk_index),
        "ego_past_shape": [int(num_samples), config.obs_len, len(NODE_FEATURE_NAMES)],
        "neighbor_past_shape": [int(num_samples), config.max_neighbors, config.obs_len, len(NODE_FEATURE_NAMES)],
        "edge_attr_shape": [int(num_samples), config.max_neighbors, len(EDGE_FEATURE_NAMES)],
        "future_shape": [int(num_samples), config.pred_len, 2],
        "coord_unit": "meter" if config.convert_feet_to_meters else "feet",
        "config": {**asdict(config), "raw_dir": str(config.raw_dir), "output_dir": str(config.output_dir)},
    }
    with (config.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare graph-based NGSIM trajectory samples.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ngsim"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/ngsim_lankershim_graph"))
    parser.add_argument("--location", type=str, default="lankershim")
    parser.add_argument("--obs-len", type=int, default=30)
    parser.add_argument("--pred-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--radius-m", type=float, default=30.0)
    parser.add_argument("--max-neighbors", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=10000)
    parser.add_argument("--keep-feet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metadata = process_ngsim_graph(
        NGSIMGraphProcessConfig(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            location=args.location,
            obs_len=args.obs_len,
            pred_len=args.pred_len,
            stride=args.stride,
            fps=args.fps,
            radius_m=args.radius_m,
            max_neighbors=args.max_neighbors,
            chunk_size=args.chunk_size,
            convert_feet_to_meters=not args.keep_feet,
        )
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
