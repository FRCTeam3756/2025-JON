import os
import math
import logging
from typing import Optional, Tuple, List
from logs.logging_setup import setup_logger

from config import AutoAlgaeConfig
from navigator.trackable_objects import Algae

################################################


class AlgaePickupCommand:
    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        setup_logger(file_name)
        self.logger = logging.getLogger(file_name)

    def get_algae_navigation_command(self, algae: Algae) -> Tuple[float, float, float, bool]:
        if not algae:
            self.logger.warning("No algae found")
            return (0.0, 0.0, 0.0, False)

        speed_percent: Optional[float] = 0.0
        x: Optional[float] = 0.0
        y: Optional[float] = 0.0
        rot: Optional[float] = 0.0

        if algae.relative_distance_mm and algae.relative_distance_mm > AutoAlgaeConfig.ALGAE_DESIRED_DISTANCE_MM:
            speed_percent = min((algae.relative_distance_mm - AutoAlgaeConfig.ALGAE_DESIRED_DISTANCE_MM) / (
                AutoAlgaeConfig.ALGAE_MAX_DISTANCE_MM - AutoAlgaeConfig.ALGAE_DESIRED_DISTANCE_MM) * 100, 100)
        else:
            speed_percent = None

        if algae.relative_angle_deg:
            angle_rad = math.radians(algae.relative_angle_deg)
            rot = max(min(algae.relative_angle_deg / 180 * 100, 100), -100)

            if speed_percent:
                x = speed_percent * math.cos(angle_rad)
                y = speed_percent * math.sin(angle_rad)

        self.logger.info(
            f"Algae navigation command: x={x:.1f}%, y={y:.1f}%, rot={rot:.1f}%")

        if x and y and rot:
            return (x, y, rot, True)
        else:
            return (0.0, 0.0, 0.0, False)

    def compute_best_algae(self, algaes: List[Algae]) -> Optional[Algae]:
        """Compute the best game piece based on weighted attributes."""
        if not algaes:
            return None

        best_piece = None
        best_score = 0.0

        for piece in algaes:
            score = self.compute_score(piece)
            if not score:
                continue
            else:
                if score > best_score:
                    best_piece = piece
                    best_score = score

        return best_piece

    def compute_score(self, algae: Algae) -> Optional[float]:
        """Calculate the weighted score for a game piece."""
        if not (algae.confidence and algae.relative_distance_mm and algae.relative_angle_deg):
            return None
        
        return (
            AutoAlgaeConfig.ALGAE_CONFIDENCE_WEIGHT_PCT * algae.confidence +
            AutoAlgaeConfig.ALGAE_DISTANCE_WEIGHT_PCT * ((120 - algae.relative_distance_mm) / 120) +
            AutoAlgaeConfig.ALGAE_ANGULAR_WEIGHT_PCT *
            (1 - abs(algae.relative_angle_deg) / 180)
        )