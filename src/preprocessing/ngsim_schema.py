from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NGSIMColumns:
    vehicle_id: str = "Vehicle_ID"
    frame_id: str = "Frame_ID"
    total_frames: str = "Total_Frames"
    global_time: str = "Global_Time"
    local_x: str = "Local_X"
    local_y: str = "Local_Y"
    global_x: str = "Global_X"
    global_y: str = "Global_Y"
    length: str = "v_Length"
    width: str = "v_Width"
    vehicle_class: str = "v_Class"
    speed: str = "v_Vel"
    acc: str = "v_Acc"
    lane_id: str = "Lane_ID"
    preceding: str = "Preceding"
    following: str = "Following"
    space_headway: str = "Space_Headway"
    time_headway: str = "Time_Headway"
    location: str = "Location"


NGSIM_COLUMNS = NGSIMColumns()

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
    "acc",
    "lane_id",
    "vehicle_class",
    "length",
    "width",
    "preceding",
    "preceding_exists",
    "leader_dx",
    "leader_dy",
    "leader_dvx",
    "leader_dvy",
    "leader_acc",
    "leader_exists",
    "following",
    "space_headway",
    "time_headway",
]
