import os
import logging
from typing import Tuple
from logs.logging_setup import setup_logger

from config import AutoCoralConfig
from decision_engine.trackable_objects import *

################################################

class CoralPickupCommand:
    REQUIRED_ATTRIBUTES = ['confidence', 'distance', 'angle']

    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        self.logger = setup_logger(file_name)

    def get_coral_navigation_command(self, coral: Coral) -> Tuple[float, float, float, bool]:
        if not coral:
            self.logger.warning("No coral found")
            return [0.0, 0.0, 0.0, False]
        
        if coral.distance > AutoCoralConfig.CORAL_DESIRED_DISTANCE_IN_MM:
            speed_percent = min((coral.distance - AutoCoralConfig.CORAL_DESIRED_DISTANCE_IN_MM) / (AutoCoralConfig.CORAL_MAX_DISTANCE_IN_MM - AutoCoralConfig.CORAL_DESIRED_DISTANCE_IN_MM) * 100, 100)
        else:
            speed_percent = 0.0

        angle_rad = math.radians(coral.angle)
        x = speed_percent * math.cos(angle_rad)
        y = speed_percent * math.sin(angle_rad)

        rot = max(min(coral.angle / 180 * 100, 100), -100)

        self.logger.info(f"Coral navigation command: x={x:.1f}%, y={y:.1f}%, rot={rot:.1f}%")
        return [x, y, rot, True]

    def compute_best_coral(self, *corals: Coral) -> Coral:
        """Compute the best game piece based on weighted attributes."""
        if not corals:
            return

        best_piece = None

        for piece in corals:
            if self.validate_coral(piece):
                if best_piece is None or self.compute_score(piece) > self.compute_score(best_piece):
                    best_piece = piece

        return best_piece

    def validate_coral(self, coral: Coral) -> Coral:
        """Check if a game piece has all required attributes."""
        missing_attributes = [attr for attr in self.REQUIRED_ATTRIBUTES if getattr(
            coral, attr, None) is None]
        if missing_attributes:
            self.logger.error(
                f"Game piece {coral} is missing attributes: {', '.join(missing_attributes)}")
            return False
        return True

    def compute_score(self, coral):
        """Calculate the weighted score for a game piece."""
        return (
            AutoCoralConfig.CORAL_CONFIDENCE_WEIGHT * coral.confidence +
            AutoCoralConfig.CORAL_DISTANCE_WEIGHT * ((120 - coral.distance) / 120) +
            AutoCoralConfig.CORAL_ANGULAR_WEIGHT * (1 - abs(coral.angle) / 180)
        )