import os
import math
import logging
from typing import Tuple
from logs.logging_setup import setup_logger

from src.apriltags.apriltags import AprilTagDetection, AprilTagFinder
from config import AutoReefConfig

################################################


class ReefScoringCommand:
    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        setup_logger(file_name)
        self.logger = logging.getLogger(file_name)

    def get_reef_navigation_command(self, reef_apriltag: AprilTagDetection) -> Tuple[float, float, float, bool]:
        if not (reef_apriltag and reef_apriltag.relative_distance and reef_apriltag.relative_angle):
            self.logger.warning("Reef not found")
            return (0.0, 0.0, 0.0, False)

        if reef_apriltag.relative_distance > AutoReefConfig.REEF_DESIRED_DISTANCE_MM:
            speed_percent = min((reef_apriltag.relative_distance - AutoReefConfig.REEF_DESIRED_DISTANCE_MM) / (
                AutoReefConfig.REEF_MAX_DISTANCE_MM - AutoReefConfig.REEF_DESIRED_DISTANCE_MM) * 100, 100)
        else:
            speed_percent = 0.0

        angle_in_radians = math.radians(reef_apriltag.relative_angle)
        x = speed_percent * math.cos(angle_in_radians)
        y = speed_percent * math.sin(angle_in_radians)

        rot = max(min(reef_apriltag.relative_angle / 180 * 100, 100), -100)

        self.logger.info(
            f"Reef navigation command: x={x:.1f}%, y={y:.1f}%, rot={rot:.1f}%")
        return (x, y, rot, True)
