import os
import math
from typing import List, Optional, Tuple

import numpy as np
from pupil_apriltags import Detection

from logs.logging_setup import setup_logger
from config import AprilTagConfig
from src.apriltags.apriltags import AprilTagDetection, AprilTagFinder
from src.odometry.odometry import Odometry

class Localization:
    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        self.logger = setup_logger(file_name)
        self.logger.info("Odometry logger initialized.")

        self.odometry = Odometry()
        self.last_position = 8, 2, 2.9

    def object_world_coords(self, relative_distance_mm: float, relative_angle_deg: float, robot_x_m: float, robot_y_m: float, robot_heading_rad: float) -> Tuple[float, float]:
        dist_m = relative_distance_mm * 0.001
        rel_angle_rad = math.radians(relative_angle_deg)
        abs_angle_rad = robot_heading_rad + rel_angle_rad
        obj_x_m = robot_x_m + dist_m * math.cos(abs_angle_rad)
        obj_y_m = robot_y_m + dist_m * math.sin(abs_angle_rad)
        return obj_x_m, obj_y_m

    def get_world_position(self, apriltag: AprilTagDetection) -> Optional[Tuple[float, float, float]]:
        apriltag_number = apriltag.id
        relative_distance_m = apriltag.relative_distance_m
        relative_angle_deg = apriltag.relative_angle_deg
        
        if apriltag_number not in AprilTagConfig.APRILTAG_POSITIONS_M:
            self.logger.error(f"Invalid AprilTag number: {apriltag_number}")
            raise ValueError(f"AprilTag {apriltag_number} not found.")
        
        if not relative_angle_deg or not relative_distance_m:
            return None

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

    def process_frame(self, apriltags: List[AprilTagDetection]) -> np.ndarray:
        if len(apriltags) > 0:
            closest_apriltag = AprilTagFinder.get_best_tag(apriltags)
            if closest_apriltag:
                first_apriltag = AprilTagDetection(closest_apriltag, 1)
                apriltags.remove(closest_apriltag)
                if len(apriltags) == 0:
                    vision_world_position = self.get_world_position(first_apriltag)
                    odometry_world_position = self.odometry.get_world_position(self.last_position)
                    
                    world_position = vision_world_position[0]
                else:
                    apriltags
                    second_closest_apriltag()
        frame = visualization.render_frame(robot_x_m, robot_y_m, robot_heading_rad)
        self.update_past_detections(robot_x_m, robot_y_m, robot_heading_rad)

        return frame