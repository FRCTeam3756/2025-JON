import os
import math
import logging
from typing import Tuple
from logs.logging_setup import setup_logger

from robotpy_apriltag import AprilTagDetection

from apriltags.apriltag_finder import AprilTagFinder
from config import AutoProcessorConfig

################################################

class ProcessorScoringCommand:
    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        setup_logger(file_name)
        self.logger = logging.getLogger(file_name)

    def get_processor_navigation_command(self, processor_apriltag: AprilTagDetection) -> Tuple[float, float, float, bool]:
        if not processor_apriltag:
            self.logger.warning("Processor not found")
            return [0.0, 0.0, 0.0, False]
        
        distance_to_apriltag = AprilTagFinder.estimate_distance(processor_apriltag)
        angular_diviation = AprilTagFinder.calculate_anglular_diviation(processor_apriltag)
        
        if distance_to_apriltag > AutoProcessorConfig.PROCESSOR_DESIRED_DISTANCE_IN_MM:
            speed_percent = min((distance_to_apriltag - AutoProcessorConfig.PROCESSOR_DESIRED_DISTANCE_IN_MM) / (AutoProcessorConfig.PROCESSOR_MAX_DISTANCE_IN_MM - AutoProcessorConfig.PROCESSOR_DESIRED_DISTANCE_IN_MM) * 100, 100)
        else:
            speed_percent = 0.0

        angle_in_radians = math.radians(angular_diviation)
        x = speed_percent * math.cos(angle_in_radians)
        y = speed_percent * math.sin(angle_in_radians)

        rot = max(min(processor_apriltag.angle / 180 * 100, 100), -100)

        self.logger.info(f"Processor navigation command: x={x:.1f}%, y={y:.1f}%, rot={rot:.1f}%")
        return [x, y, rot, True]