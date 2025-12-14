"""
Copyright (c) FRC Team 3756 RamFerno.
Open Source Software; you can modify and/or share it under the terms of
the license viewable in the root directory of this project.
"""

import os
import math
import logging
from typing import Tuple
from logs.logging_setup import setup_logger

from src.apriltags.apriltags import AprilTagDetection, AprilTagFinder
from constants.field.processor import ProcessorConfig

################################################


class ProcessorScoringCommand:
    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        setup_logger(file_name)
        self.logger = logging.getLogger(file_name)

    def get_processor_navigation_command(self, processor_apriltag: AprilTagDetection) -> Tuple[float, float, float, bool]:
        if not (processor_apriltag and processor_apriltag.relative_distance_m and processor_apriltag.relative_angle_deg):
            self.logger.warning("Processor not found")
            return (0.0, 0.0, 0.0, False)

        if processor_apriltag.relative_distance_m > ProcessorConfig.PROCESSOR_DESIRED_DISTANCE_MM:
            speed_percent = min((processor_apriltag.relative_distance_m - ProcessorConfig.PROCESSOR_DESIRED_DISTANCE_MM) / (
                ProcessorConfig.PROCESSOR_MAX_DISTANCE_MM - ProcessorConfig.PROCESSOR_DESIRED_DISTANCE_MM) * 100, 100)
        else:
            speed_percent = 0.0

        angle_in_radians = math.radians(processor_apriltag.relative_angle_deg)
        x = speed_percent * math.cos(angle_in_radians)
        y = speed_percent * math.sin(angle_in_radians)

        rot = max(min(processor_apriltag.relative_angle_deg / 180 * 100, 100), -100)

        self.logger.info(
            f"Processor navigation command: x={x:.1f}%, y={y:.1f}%, rot={rot:.1f}%")
        return (x, y, rot, True)
