import os
import math
from typing import Tuple

from logs.logging_setup import setup_logger
from config import AprilTagConfig

class Localization:
    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        self.logger = setup_logger(file_name)
        self.logger.info("Odometry logger initialized.")

    def object_world_coords(self, relative_distance_mm: float, relative_angle_deg: float, robot_x_m: float, robot_y_m: float, robot_heading_rad: float) -> Tuple[float, float]:
        dist_m = relative_distance_mm * 0.001
        rel_angle_rad = math.radians(relative_angle_deg)
        abs_angle_rad = robot_heading_rad + rel_angle_rad
        obj_x_m = robot_x_m + dist_m * math.cos(abs_angle_rad)
        obj_y_m = robot_y_m + dist_m * math.sin(abs_angle_rad)
        return obj_x_m, obj_y_m

    def get_world_position(self, apriltag_number: int, relative_distance_m: float, relative_angle_deg: float) -> Tuple[float, float, float]:
        if apriltag_number not in AprilTagConfig.APRILTAG_POSITIONS_M:
            self.logger.error(f"Invalid AprilTag number: {apriltag_number}")
            raise ValueError(f"AprilTag {apriltag_number} not found.")

        tag_x_m, tag_y_m, tag_rotation_deg = AprilTagConfig.APRILTAG_POSITIONS_M[apriltag_number]

        tag_heading_rad = math.radians(tag_rotation_deg)
        rel_angle_rad = math.radians(relative_angle_deg)

        robot_rel_x = -relative_distance_m * math.cos(rel_angle_rad)
        robot_rel_y = relative_distance_m * math.sin(rel_angle_rad)

        robot_x_m = tag_x_m - (robot_rel_x * math.cos(tag_heading_rad) - robot_rel_y * math.sin(tag_heading_rad))
        robot_y_m = tag_y_m - (robot_rel_x * math.sin(tag_heading_rad) + robot_rel_y * math.cos(tag_heading_rad))

        robot_heading_rad = tag_heading_rad + rel_angle_rad - math.pi
        robot_heading_rad = (robot_heading_rad + math.pi) % (2 * math.pi) - math.pi

        self.logger.info(f'RamFerno sees AprilTag {apriltag_number} {relative_distance_m} away from it at {relative_angle_deg} from its center.')
        self.logger.info(f'The apriltag is absolutely positioned at {tag_x_m}x, {tag_y_m}y, facing {tag_rotation_deg} degrees in the real world.')
        self.logger.info(f'It thinks the robot is at {robot_x_m}x, {robot_y_m}y, facing {robot_heading_rad} radians')
        return robot_x_m, robot_y_m, robot_heading_rad