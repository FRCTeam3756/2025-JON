import os
import logging
from typing import Optional, Tuple
from logs.logging_setup import setup_logger

from config import AutoAlgaeConfig
from decision_engine.trackable_objects import *

################################################

class AlgaePickupCommand:
    REQUIRED_ATTRIBUTES = ['confidence', 'distance', 'angle']

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
        
        if algae.distance_in_mm and algae.distance_in_mm > AutoAlgaeConfig.ALGAE_DESIRED_DISTANCE_IN_MM:
            speed_percent = min((algae.distance_in_mm - AutoAlgaeConfig.ALGAE_DESIRED_DISTANCE_IN_MM) / (AutoAlgaeConfig.ALGAE_MAX_DISTANCE_IN_MM - AutoAlgaeConfig.ALGAE_DESIRED_DISTANCE_IN_MM) * 100, 100)
        else:
            speed_percent = None
        
        if algae.angle_in_degrees:
            angle_rad = math.radians(algae.angle_in_degrees)
            rot = max(min(algae.angle_in_degrees / 180 * 100, 100), -100)

            if speed_percent:
                x = speed_percent * math.cos(angle_rad)
                y = speed_percent * math.sin(angle_rad)

        self.logger.info(f"Algae navigation command: x={x:.1f}%, y={y:.1f}%, rot={rot:.1f}%")
        
        if x and y and rot:
            return (x, y, rot, True)
        else:
            return (0.0, 0.0, 0.0, False)
        

    def compute_best_algae(self, *algaes: Algae) -> Optional[Algae]:
        """Compute the best game piece based on weighted attributes."""
        if not algaes:
            return None

        best_piece = None

        for piece in algaes:
            if self.validate_algae(piece):
                if best_piece is None or self.compute_score(piece) > self.compute_score(best_piece):
                    best_piece = piece

        return best_piece

    def validate_algae(self, algae: Algae) -> bool:
        """Check if a game piece has all required attributes."""
        missing_attributes = [attr for attr in self.REQUIRED_ATTRIBUTES if getattr(
            algae, attr, None) is None]
        if missing_attributes:
            self.logger.error(
                f"Game piece {algae} is missing attributes: {', '.join(missing_attributes)}")
            return False
        return True

    def compute_score(self, algae: Algae) -> float:
        """Calculate the weighted score for a game piece."""
        if algae.confidence and algae.distance_in_mm and algae.angle_in_degrees:
            return (
                AutoAlgaeConfig.ALGAE_CONFIDENCE_WEIGHT * algae.confidence +
                AutoAlgaeConfig.ALGAE_DISTANCE_WEIGHT * ((120 - algae.distance_in_mm) / 120) +
                AutoAlgaeConfig.ALGAE_ANGULAR_WEIGHT * (1 - abs(algae.angle_in_degrees) / 180)
            )
        else:
            return 0.0