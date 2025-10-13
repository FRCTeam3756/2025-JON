import os
import math
from typing import Tuple

from config import FieldConfig
from logs.logging_setup import setup_logger


class Localization:
    INCH_TO_M = 0.0254

    APRILTAG_POSITIONS = {
        1: (656.98, 24.73, 126),
        2: (656.98, 291.90, 234),
        3: (452.40, 316.21, 270),
        4: (365.20, 241.44, 0),
        5: (365.20, 75.19, 0),
        6: (530.49, 129.97, 300),
        7: (546.87, 158.30, 0),
        8: (530.49, 186.63, 60),
        9: (497.77, 186.63, 120),
        10: (481.39, 158.30, 180),
        11: (497.77, 129.97, 240),
        12: (33.91, 24.73, 54),
        13: (33.91, 291.90, 306),
        14: (325.68, 241.44, 180),
        15: (325.68, 75.19, 180),
        16: (238.49, 0.42, 90),
        17: (160.39, 129.97, 240),
        18: (144.00, 158.30, 180),
        19: (160.39, 186.63, 120),
        20: (193.10, 186.63, 60),
        21: (209.49, 158.30, 0),
        22: (193.10, 129.97, 300),
    }

    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        self.logger = setup_logger(file_name)
        self.logger.info("Odometry logger initialized.")

        self.target_x = FieldConfig.FIELD_WIDTH_M * 0.85
        self.target_y = FieldConfig.FIELD_HEIGHT_M * 0.75
        self.APRILTAG_POSITIONS = {
            k: (x * self.INCH_TO_M, y * self.INCH_TO_M, r)
            for k, (x, y, r) in self.APRILTAG_POSITIONS.items()
        }

    def object_world_coords(self, relative_distance_mm: float, relative_angle_deg: float, robot_x_m: float, robot_y_m: float, robot_heading_rad: float) -> Tuple[float, float]:
        dist_m = relative_distance_mm * 0.001
        rel_angle_rad = math.radians(relative_angle_deg)
        abs_angle_rad = robot_heading_rad + rel_angle_rad
        obj_x_m = robot_x_m + dist_m * math.cos(abs_angle_rad)
        obj_y_m = robot_y_m + dist_m * math.sin(abs_angle_rad)
        return obj_x_m, obj_y_m

    def get_world_position(self, apriltag_number: int, relative_distance_mm: float, relative_angle_deg: float) -> Tuple[float, float, float]:
        if apriltag_number not in self.APRILTAG_POSITIONS:
            self.logger.error(f"Invalid AprilTag number: {apriltag_number}")
            raise ValueError(f"AprilTag {apriltag_number} not found.")

        tag_x_m, tag_y_m, tag_rotation_deg = self.APRILTAG_POSITIONS[apriltag_number]

        dist_m = relative_distance_mm * 0.001
        tag_heading_rad = math.radians(tag_rotation_deg)
        rel_angle_rad = math.radians(relative_angle_deg)

        robot_heading_rad = tag_heading_rad - rel_angle_rad

        tag_rel_x = dist_m * math.cos(rel_angle_rad)
        tag_rel_y = dist_m * math.sin(rel_angle_rad)

        robot_x_m = tag_x_m - (math.cos(tag_heading_rad) * tag_rel_x - math.sin(tag_heading_rad) * tag_rel_y)
        robot_y_m = tag_y_m - (math.sin(tag_heading_rad) * tag_rel_x + math.cos(tag_heading_rad) * tag_rel_y)

        return robot_x_m, robot_y_m, robot_heading_rad