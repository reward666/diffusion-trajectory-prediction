from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.preprocessing.ngsim_schema import NGSIM_COLUMNS, STANDARD_COLUMNS


@dataclass(frozen=True)
class NGSIMProcessConfig:
    raw_dir: Path
    output_dir: Path
    obs_len: int = 30
    pred_len: int = 50
    stride: int = 5
    fps: float = 10.0
    use_global_xy: bool = False
    convert_feet_to_meters: bool = True
    min_track_len: int | None = None
    chunk_size: int = 10000
    location: str | None = None

    @property
    def total_len(self) -> int:
        return self.obs_len + self.pred_len


def find_ngsim_files(raw_dir: Path) -> list[Path]:
    suffixes = {".csv", ".txt"}
    return sorted(path for path in raw_dir.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def infer_scene_id(path: Path, raw_dir: Path) -> str:
    rel = path.relative_to(raw_dir)
    if len(rel.parts) > 1:
        return rel.parts[0]
    stem = path.stem.lower()
    if "us101" in stem or "us-101" in stem:
        return "us101"
    if "i80" in stem or "i-80" in stem:
        return "i80"
    return path.stem


def read_ngsim_table(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    return pd.read_csv(path, sep=r"\s+", engine="python")


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(col).strip() for col in df.columns]
    return df


def _require_columns(df: pd.DataFrame, path: Path) -> None:
    cols = NGSIM_COLUMNS
    required = [cols.vehicle_id, cols.frame_id, cols.local_x, cols.local_y]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing required NGSIM columns: {missing}")


def _numeric_or_default(df: pd.DataFrame, column: str, default: float = np.nan) -> pd.Series:
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce")
    return pd.Series(default, index=df.index, dtype="float64")


def load_ngsim_file(path: Path, raw_dir: Path, config: NGSIMProcessConfig) -> pd.DataFrame:
    df = _clean_columns(read_ngsim_table(path))
    _require_columns(df, path)

    cols = NGSIM_COLUMNS
    if config.use_global_xy and cols.global_x in df.columns and cols.global_y in df.columns:
        x_col, y_col = cols.global_x, cols.global_y
    else:
        x_col, y_col = cols.local_x, cols.local_y

    if cols.location in df.columns:
        scene_id = df[cols.location].astype(str).str.strip().str.lower()
        scene_id = scene_id.replace({"": infer_scene_id(path, config.raw_dir), "nan": infer_scene_id(path, config.raw_dir)})
    else:
        scene_id = infer_scene_id(path, config.raw_dir)

    out = pd.DataFrame(
        {
            "scene_id": scene_id,
            "source_file": str(path.relative_to(raw_dir)),
            "frame": pd.to_numeric(df[cols.frame_id], errors="coerce").astype("Int64"),
            "track_id": df[cols.vehicle_id].astype(str),
            "x": pd.to_numeric(df[x_col], errors="coerce"),
            "y": pd.to_numeric(df[y_col], errors="coerce"),
            "speed": _numeric_or_default(df, cols.speed),
            "acc": _numeric_or_default(df, cols.acc),
            "lane_id": _numeric_or_default(df, cols.lane_id),
            "vehicle_class": _numeric_or_default(df, cols.vehicle_class),
            "length": _numeric_or_default(df, cols.length),
            "width": _numeric_or_default(df, cols.width),
            "preceding": _numeric_or_default(df, cols.preceding),
            "following": _numeric_or_default(df, cols.following),
            "space_headway": _numeric_or_default(df, cols.space_headway),
            "time_headway": _numeric_or_default(df, cols.time_headway),
        }
    )

    out = out.dropna(subset=["frame", "x", "y"]).copy()
    out["frame"] = out["frame"].astype(int)

    if config.convert_feet_to_meters:
        ft_cols = ["x", "y", "speed", "acc", "length", "width", "space_headway"]
        for col in ft_cols:
            out[col] = out[col] * 0.3048

    if cols.global_time in df.columns:
        global_time = pd.to_numeric(df.loc[out.index, cols.global_time], errors="coerce")
        if global_time.notna().any():
            out["t"] = global_time / 1000.0
        else:
            out["t"] = out["frame"] / config.fps
    else:
        out["t"] = out["frame"] / config.fps

    out = out.sort_values(["scene_id", "source_file", "track_id", "frame"])
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

    out["preceding_exists"] = (out["preceding"].fillna(0) > 0).astype(np.float32)
    return out[STANDARD_COLUMNS]


def load_ngsim_tracks(config: NGSIMProcessConfig) -> tuple[pd.DataFrame, dict]:
    files = find_ngsim_files(config.raw_dir)
    if not files:
        raise FileNotFoundError(f"No .csv or .txt NGSIM files found under {config.raw_dir}.")

    frames = []
    skipped = []
    for path in files:
        try:
            frames.append(load_ngsim_file(path, config.raw_dir, config))
        except ValueError as exc:
            skipped.append({"file": str(path), "reason": str(exc)})

    if not frames:
        raise ValueError("No valid NGSIM trajectory files were loaded.")

    tracks = pd.concat(frames, ignore_index=True)
    metadata = {
        "dataset": "ngsim",
        "raw_dir": str(config.raw_dir),
        "num_files": len(files),
        "num_loaded_files": len(frames),
        "num_skipped_files": len(skipped),
        "skipped_files": skipped,
        "num_rows": int(len(tracks)),
        "num_tracks": int(tracks[["source_file", "track_id"]].drop_duplicates().shape[0]),
        "coord_unit": "meter" if config.convert_feet_to_meters else "feet",
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


def build_prediction_windows(tracks: pd.DataFrame, config: NGSIMProcessConfig) -> dict[str, np.ndarray]:
    past, future, meta_rows = [], [], []
    feature_cols = [
        "x",
        "y",
        "vx",
        "vy",
        "speed",
        "acc",
        "lane_id",
        "vehicle_class",
        "length",
        "width",
        "preceding_exists",
        "space_headway",
        "time_headway",
    ]
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
        raise ValueError("No valid fixed-length NGSIM trajectory windows were created.")

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


def _save_sample_chunk(
    samples_dir: Path,
    chunk_index: int,
    past: list[np.ndarray],
    future: list[np.ndarray],
    meta_rows: list[tuple],
    feature_cols: list[str],
) -> int:
    if not past:
        return chunk_index
    samples_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        samples_dir / f"chunk_{chunk_index:05d}.npz",
        past=np.stack(past),
        future=np.stack(future),
        meta=np.array(meta_rows, dtype=[
            ("scene_id", "U128"),
            ("source_file", "U256"),
            ("track_id", "U64"),
            ("obs_start_frame", "i8"),
            ("obs_end_frame", "i8"),
            ("pred_end_frame", "i8"),
        ]),
        feature_names=np.array(feature_cols),
    )
    return chunk_index + 1


def process_ngsim(config: NGSIMProcessConfig) -> dict:
    files = find_ngsim_files(config.raw_dir)
    if not files:
        raise FileNotFoundError(f"No .csv or .txt NGSIM files found under {config.raw_dir}.")

    config.output_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = config.output_dir / "samples_chunks"
    if samples_dir.exists() and any(samples_dir.glob("chunk_*.npz")):
        raise FileExistsError(
            f"{samples_dir} already contains sample chunks. "
            "Move or remove that directory before starting a new processing run."
        )

    feature_cols = [
        "x", "y", "vx", "vy", "speed", "acc", "lane_id", "vehicle_class",
        "length", "width", "preceding_exists", "space_headway", "time_headway",
    ]
    tracks_path = config.output_dir / "tracks.csv"
    skipped = []
    num_rows = 0
    num_tracks = 0
    num_samples = 0
    chunk_index = 0
    wrote_tracks_header = False
    past_buffer: list[np.ndarray] = []
    future_buffer: list[np.ndarray] = []
    meta_buffer: list[tuple] = []

    for file_index, path in enumerate(files, start=1):
        print(f"[{file_index}/{len(files)}] loading {path}", flush=True)
        try:
            tracks = load_ngsim_file(path, config.raw_dir, config)
        except ValueError as exc:
            skipped.append({"file": str(path), "reason": str(exc)})
            print(f"[{file_index}/{len(files)}] skipped: {exc}", flush=True)
            continue

        if config.location:
            location = config.location.strip().lower()
            tracks = tracks[tracks["scene_id"] == location].copy()
            print(f"[{file_index}/{len(files)}] filtered location={location}: {len(tracks)} rows", flush=True)
            if tracks.empty:
                continue

        tracks.to_csv(tracks_path, mode="a" if wrote_tracks_header else "w", header=not wrote_tracks_header, index=False)
        wrote_tracks_header = True
        num_rows += len(tracks)
        num_tracks += int(tracks[["scene_id", "source_file", "track_id"]].drop_duplicates().shape[0])

        min_len = config.min_track_len or config.total_len
        group_cols = ["scene_id", "source_file", "track_id"]
        for (scene_id, source_file, track_id), track in tracks.groupby(group_cols, sort=False):
            if len(track) < min_len:
                continue
            for window in iter_track_windows(track, config.total_len, config.stride):
                obs = window.iloc[: config.obs_len]
                pred = window.iloc[config.obs_len :]
                past_buffer.append(obs[feature_cols].fillna(-1.0).to_numpy(dtype=np.float32))
                future_buffer.append(pred[["x", "y"]].to_numpy(dtype=np.float32))
                meta_buffer.append((
                    scene_id,
                    source_file,
                    track_id,
                    int(obs["frame"].iloc[0]),
                    int(obs["frame"].iloc[-1]),
                    int(pred["frame"].iloc[-1]),
                ))
                num_samples += 1

                if len(past_buffer) >= config.chunk_size:
                    chunk_index = _save_sample_chunk(
                        samples_dir, chunk_index, past_buffer, future_buffer, meta_buffer, feature_cols
                    )
                    print(f"  wrote chunk_{chunk_index - 1:05d}.npz ({num_samples} samples total)", flush=True)
                    past_buffer.clear()
                    future_buffer.clear()
                    meta_buffer.clear()

        del tracks

    chunk_index = _save_sample_chunk(samples_dir, chunk_index, past_buffer, future_buffer, meta_buffer, feature_cols)
    if num_samples == 0:
        raise ValueError("No valid fixed-length NGSIM trajectory windows were created.")

    metadata = {
        "dataset": "ngsim",
        "raw_dir": str(config.raw_dir),
        "samples_dir": str(samples_dir),
        "num_files": len(files),
        "num_loaded_files": len(files) - len(skipped),
        "num_skipped_files": len(skipped),
        "skipped_files": skipped,
        "num_rows": int(num_rows),
        "num_tracks": int(num_tracks),
        "num_samples": int(num_samples),
        "num_chunks": int(chunk_index),
        "past_shape": [int(num_samples), config.obs_len, len(feature_cols)],
        "future_shape": [int(num_samples), config.pred_len, 2],
        "coord_unit": "meter" if config.convert_feet_to_meters else "feet",
        "config": {
            **asdict(config),
            "raw_dir": str(config.raw_dir),
            "output_dir": str(config.output_dir),
        },
    }
    with (config.output_dir / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare NGSIM trajectories for trajectory prediction.")
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/ngsim"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/ngsim"))
    parser.add_argument("--obs-len", type=int, default=30)
    parser.add_argument("--pred-len", type=int, default=50)
    parser.add_argument("--stride", type=int, default=5)
    parser.add_argument("--fps", type=float, default=10.0)
    parser.add_argument("--global-xy", action="store_true", help="Use Global_X/Global_Y instead of Local_X/Local_Y.")
    parser.add_argument("--keep-feet", action="store_true", help="Keep NGSIM distance units in feet.")
    parser.add_argument("--chunk-size", type=int, default=10000, help="Number of trajectory windows per output chunk.")
    parser.add_argument("--location", type=str, help="Only process one Location value, for example us-101 or i-80.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = NGSIMProcessConfig(
        raw_dir=args.raw_dir,
        output_dir=args.output_dir,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        stride=args.stride,
        fps=args.fps,
        use_global_xy=args.global_xy,
        convert_feet_to_meters=not args.keep_feet,
        chunk_size=args.chunk_size,
        location=args.location,
    )
    metadata = process_ngsim(config)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
