from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.preprocessing.citysim_schema import CITYSIM_COLUMNS, STANDARD_COLUMNS


@dataclass(frozen=True)
class CitySimProcessConfig:
    raw_dir: Path
    output_dir: Path
    obs_len: int = 30
    pred_len: int = 60
    stride: int = 10
    fps: float = 30.0
    use_metric: bool = True
    min_track_len: int | None = None

    @property
    def total_len(self) -> int:
        return self.obs_len + self.pred_len


def find_citysim_csvs(raw_dir: Path) -> list[Path]:
    csvs = sorted(raw_dir.rglob("*.csv"))
    return [path for path in csvs if "trajectories" in str(path.parent).lower()]


def infer_scene_id(csv_path: Path, raw_dir: Path) -> str:
    rel = csv_path.relative_to(raw_dir)
    parts = list(rel.parts)
    if len(parts) >= 3 and parts[-2].lower() == "trajectories":
        return parts[-3]
    return csv_path.stem


def _choose_xy_columns(df: pd.DataFrame, use_metric: bool) -> tuple[str, str, str]:
    cols = CITYSIM_COLUMNS
    if use_metric and cols.x_feet in df.columns and cols.y_feet in df.columns:
        return cols.x_feet, cols.y_feet, "feet"
    if cols.x_pixel in df.columns and cols.y_pixel in df.columns:
        return cols.x_pixel, cols.y_pixel, "pixel"
    raise ValueError(
        "CitySim CSV is missing center coordinate columns. Expected either "
        f"({cols.x_feet}, {cols.y_feet}) or ({cols.x_pixel}, {cols.y_pixel})."
    )


def load_citysim_csv(csv_path: Path, raw_dir: Path, fps: float, use_metric: bool) -> tuple[pd.DataFrame, str]:
    df = pd.read_csv(csv_path)
    cols = CITYSIM_COLUMNS
    required = [cols.frame, cols.agent_id]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {missing}")

    x_col, y_col, coord_unit = _choose_xy_columns(df, use_metric)
    out = pd.DataFrame(
        {
            "scene_id": infer_scene_id(csv_path, raw_dir),
            "source_file": str(csv_path.relative_to(raw_dir)),
            "frame": pd.to_numeric(df[cols.frame], errors="coerce").astype("Int64"),
            "track_id": df[cols.agent_id].astype(str),
            "x": pd.to_numeric(df[x_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce"),
        }
    )

    if coord_unit == "feet":
        out["x"] = out["x"] * 0.3048
        out["y"] = out["y"] * 0.3048
        coord_unit = "meter"

    if cols.speed_mph in df.columns:
        out["speed"] = pd.to_numeric(df[cols.speed_mph], errors="coerce") * 0.44704
    else:
        out["speed"] = np.nan

    heading_col = cols.heading_deg if cols.heading_deg in df.columns else cols.course_deg
    if heading_col in df.columns:
        out["heading"] = np.deg2rad(pd.to_numeric(df[heading_col], errors="coerce"))
    else:
        out["heading"] = np.nan

    if cols.lane_id in df.columns:
        out["lane_id"] = pd.to_numeric(df[cols.lane_id], errors="coerce")
    else:
        out["lane_id"] = np.nan

    out = out.dropna(subset=["frame", "x", "y"]).copy()
    out["frame"] = out["frame"].astype(int)
    out = out.sort_values(["scene_id", "source_file", "track_id", "frame"])
    out["t"] = out["frame"] / fps

    grouped = out.groupby(["scene_id", "source_file", "track_id"], sort=False)
    dt = grouped["t"].diff()
    out["vx"] = grouped["x"].diff() / dt
    out["vy"] = grouped["y"].diff() / dt
    out[["vx", "vy"]] = out[["vx", "vy"]].replace([np.inf, -np.inf], np.nan)
    out[["vx", "vy"]] = grouped[["vx", "vy"]].transform(lambda s: s.bfill().ffill())
    out[["vx", "vy"]] = out[["vx", "vy"]].fillna(0.0)

    if out["speed"].isna().all():
        out["speed"] = np.hypot(out["vx"], out["vy"])
    else:
        out["speed"] = out["speed"].fillna(np.hypot(out["vx"], out["vy"]))

    if out["heading"].isna().all():
        out["heading"] = np.arctan2(out["vy"], out["vx"])
    else:
        out["heading"] = out["heading"].fillna(np.arctan2(out["vy"], out["vx"]))

    return out[STANDARD_COLUMNS], coord_unit


def load_citysim_tracks(config: CitySimProcessConfig) -> tuple[pd.DataFrame, dict]:
    csvs = find_citysim_csvs(config.raw_dir)
    if not csvs:
        raise FileNotFoundError(
            f"No CitySim trajectory CSV files found under {config.raw_dir}. "
            "Expected files inside folders named Trajectories."
        )

    frames = []
    coord_units = set()
    for csv_path in csvs:
        df, coord_unit = load_citysim_csv(csv_path, config.raw_dir, config.fps, config.use_metric)
        frames.append(df)
        coord_units.add(coord_unit)

    tracks = pd.concat(frames, ignore_index=True)
    metadata = {
        "dataset": "citysim",
        "raw_dir": str(config.raw_dir),
        "num_files": len(csvs),
        "num_rows": int(len(tracks)),
        "num_tracks": int(tracks[["source_file", "track_id"]].drop_duplicates().shape[0]),
        "coord_units": sorted(coord_units),
        "config": {
            **asdict(config),
            "raw_dir": str(config.raw_dir),
            "output_dir": str(config.output_dir),
        },
    }
    return tracks, metadata


def iter_track_windows(track: pd.DataFrame, total_len: int, stride: int) -> Iterable[pd.DataFrame]:
    track = track.sort_values("frame").reset_index(drop=True)
    frames = track["frame"].to_numpy()
    if len(track) < total_len:
        return

    for start in range(0, len(track) - total_len + 1, stride):
        end = start + total_len
        window = track.iloc[start:end]
        expected = np.arange(frames[start], frames[start] + total_len)
        if np.array_equal(frames[start:end], expected):
            yield window


def build_prediction_windows(tracks: pd.DataFrame, config: CitySimProcessConfig) -> dict[str, np.ndarray]:
    past, future, meta_rows = [], [], []
    feature_cols = ["x", "y", "vx", "vy", "speed", "heading", "lane_id"]
    min_len = config.min_track_len or config.total_len

    group_cols = ["scene_id", "source_file", "track_id"]
    for (scene_id, source_file, track_id), track in tracks.groupby(group_cols, sort=False):
        if len(track) < min_len:
            continue
        for window in iter_track_windows(track, config.total_len, config.stride):
            obs = window.iloc[: config.obs_len]
            pred = window.iloc[config.obs_len :]
            past.append(obs[feature_cols].fillna(-1.0).to_numpy(dtype=np.float32))
            future.append(pred[["x", "y"]].to_numpy(dtype=np.float32))
            meta_rows.append(
                (
                    scene_id,
                    source_file,
                    track_id,
                    int(obs["frame"].iloc[0]),
                    int(obs["frame"].iloc[-1]),
                    int(pred["frame"].iloc[-1]),
                )
            )

    if not past:
        raise ValueError("No valid fixed-length trajectory windows were created.")

    meta_dtype = [
        ("scene_id", "U128"),
        ("source_file", "U256"),
        ("track_id", "U64"),
        ("obs_start_frame", "i8"),
        ("obs_end_frame", "i8"),
        ("pred_end_frame", "i8"),
    ]
    return {
        "past": np.stack(past),
        "future": np.stack(future),
        "meta": np.array(meta_rows, dtype=meta_dtype),
        "feature_names": np.array(feature_cols),
    }


def save_outputs(tracks: pd.DataFrame, samples: dict[str, np.ndarray], metadata: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    tracks.to_csv(output_dir / "tracks.csv", index=False)
    np.savez_compressed(output_dir / "samples.npz", **samples)
    with (output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def process_citysim(config: CitySimProcessConfig) -> dict:
    tracks, metadata = load_citysim_tracks(config)
    samples = build_prediction_windows(tracks, config)
    metadata["num_samples"] = int(samples["past"].shape[0])
    metadata["past_shape"] = list(samples["past"].shape)
    metadata["future_shape"] = list(samples["future"].shape)
    save_outputs(tracks, samples, metadata, config.output_dir)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare CitySim trajectories for trajectory prediction.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/citysim"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/citysim"))
    parser.add_argument("--obs-len", type=int, default=30)
    parser.add_argument("--pred-len", type=int, default=60)
    parser.add_argument("--stride", type=int, default=10)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--pixel", action="store_true", help="Use pixel coordinates instead of metric feet columns.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = CitySimProcessConfig(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        stride=args.stride,
        fps=args.fps,
        use_metric=not args.pixel,
    )
    metadata = process_citysim(config)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

