from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RuleBasedScorerConfig:
    fps: float = 10.0
    initial_velocity_weight: float = 1.0
    acceleration_weight: float = 0.2
    jerk_weight: float = 0.05
    lateral_weight: float = 0.2
    leader_risk_weight: float = 2.0
    safe_leader_gap_m: float = 5.0


def _feature_index(feature_names: list[str], name: str) -> int:
    if name not in feature_names:
        raise ValueError(f"Missing scorer feature: {name}")
    return feature_names.index(name)


def score_trajectory_candidates(
    predictions_xy: np.ndarray,
    past_xy: np.ndarray,
    past_features: np.ndarray,
    feature_names: list[str],
    config: RuleBasedScorerConfig,
) -> dict[str, np.ndarray]:
    if predictions_xy.ndim != 4:
        raise ValueError(f"Expected predictions shape [batch, k, pred_len, 2], got {predictions_xy.shape}.")
    if past_xy.ndim != 3:
        raise ValueError(f"Expected past_xy shape [batch, obs_len, 2], got {past_xy.shape}.")

    fps = config.fps
    current_xy = past_xy[:, -1:, :]
    candidate_start = np.repeat(current_xy[:, None, :, :], predictions_xy.shape[1], axis=1)
    prediction_steps = np.diff(np.concatenate([candidate_start, predictions_xy], axis=2), axis=2)
    prediction_velocity = prediction_steps * fps
    prediction_acceleration = np.diff(prediction_velocity, axis=2) * fps
    prediction_jerk = np.diff(prediction_acceleration, axis=2) * fps

    vx_index = _feature_index(feature_names, "vx")
    vy_index = _feature_index(feature_names, "vy")
    ego_velocity = past_features[:, -1, [vx_index, vy_index]]
    initial_velocity_penalty = np.linalg.norm(prediction_velocity[:, :, 0, :] - ego_velocity[:, None, :], axis=-1)
    acceleration_penalty = np.linalg.norm(prediction_acceleration, axis=-1).mean(axis=-1)
    jerk_penalty = np.linalg.norm(prediction_jerk, axis=-1).mean(axis=-1)
    lateral_penalty = np.abs(predictions_xy[:, :, -1, 0] - current_xy[:, None, 0, 0])

    leader_exists_index = _feature_index(feature_names, "leader_exists")
    leader_dx_index = _feature_index(feature_names, "leader_dx")
    leader_dy_index = _feature_index(feature_names, "leader_dy")
    leader_dvx_index = _feature_index(feature_names, "leader_dvx")
    leader_dvy_index = _feature_index(feature_names, "leader_dvy")
    leader_exists = past_features[:, -1, leader_exists_index]
    leader_position = past_features[:, -1, [leader_dx_index, leader_dy_index]]
    leader_velocity = ego_velocity + past_features[:, -1, [leader_dvx_index, leader_dvy_index]]
    seconds = np.arange(1, predictions_xy.shape[2] + 1, dtype=np.float32) / fps
    projected_leader = current_xy + leader_position[:, None, :] + leader_velocity[:, None, :] * seconds[None, :, None]
    leader_gap = np.linalg.norm(predictions_xy - projected_leader[:, None, :, :], axis=-1)
    unsafe_gap = np.maximum(config.safe_leader_gap_m - leader_gap, 0.0)
    leader_risk_penalty = unsafe_gap.mean(axis=-1) * leader_exists[:, None]

    total = (
        config.initial_velocity_weight * initial_velocity_penalty
        + config.acceleration_weight * acceleration_penalty
        + config.jerk_weight * jerk_penalty
        + config.lateral_weight * lateral_penalty
        + config.leader_risk_weight * leader_risk_penalty
    )
    return {
        "total": total,
        "initial_velocity": initial_velocity_penalty,
        "acceleration": acceleration_penalty,
        "jerk": jerk_penalty,
        "lateral": lateral_penalty,
        "leader_risk": leader_risk_penalty,
    }


def select_best_candidates(predictions_xy: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    best_indices = scores.argmin(axis=1)
    selected = predictions_xy[np.arange(len(predictions_xy)), best_indices]
    return selected, best_indices
