from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CitySimColumns:
    frame: str = "frameNum"
    agent_id: str = "carId"
    x_pixel: str = "carCenterX"
    y_pixel: str = "carCenterY"
    x_feet: str = "carCenterXft"
    y_feet: str = "carCenterYft"
    speed_mph: str = "speed"
    heading_deg: str = "heading"
    course_deg: str = "course"
    lane_id: str = "laneId"


CITYSIM_COLUMNS = CitySimColumns()

STANDARD_COLUMNS = [
    "scene_id",
    "source_file",
    "frame",
    "track_id",
    "t",
    "x",
    "y",
    "vx",
    "vy",
    "speed",
    "heading",
    "lane_id",
]

