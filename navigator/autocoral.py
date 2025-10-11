import os
import math
from typing import Optional, Tuple, List
from logs.logging_setup import setup_logger

from config import AutoCoralConfig
from navigator.trackable_objects import Coral

################################################


class CoralPickupCommand:
    REQUIRED_ATTRIBUTES = ['confidence',
                           'relative_distance_mm', 'relative_angle_deg']

    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        self.logger = setup_logger(file_name)

    def get_coral_navigation_command(self, coral: Coral) -> Tuple[float, float, float, bool]:
        if not coral:
            self.logger.warning("No coral found")
            return (0.0, 0.0, 0.0, False)

        speed_percent: Optional[float] = 0.0
        x: Optional[float] = 0.0
        y: Optional[float] = 0.0
        rot: Optional[float] = 0.0

        if coral.relative_distance_mm and coral.relative_distance_mm > AutoCoralConfig.CORAL_DESIRED_DISTANCE_MM:
            speed_percent = min((coral.relative_distance_mm - AutoCoralConfig.CORAL_DESIRED_DISTANCE_MM) / (
                AutoCoralConfig.CORAL_MAX_DISTANCE_MM - AutoCoralConfig.CORAL_DESIRED_DISTANCE_MM) * 100, 100)
        else:
            speed_percent = None

        if coral.relative_angle_deg:
            angle_rad = math.radians(coral.relative_angle_deg)
            rot = max(min(coral.relative_angle_deg / 180 * 100, 100), -100)

            if speed_percent:
                x = speed_percent * math.cos(angle_rad)
                y = speed_percent * math.sin(angle_rad)

        self.logger.info(
            f"Coral navigation command: x={x:.1f}%, y={y:.1f}%, rot={rot:.1f}%")
        return (x, y, rot, True)

    def compute_best_coral(self, corals: List[Coral]) -> Optional[Coral]:
        """Compute the best game piece based on weighted attributes."""
        if not corals:
            return None

        best_piece = None
        best_score = 0.0

        for piece in corals:
            if self.validate_coral(piece):
                score = self.compute_score(piece)
                if not score:
                    continue
                else:
                    if score > best_score:
                        best_piece = piece
                        best_score = score

        return best_piece

    def validate_coral(self, coral: Coral) -> bool:
        """Check if a game piece has all required attributes."""
        missing_attributes = [attr for attr in self.REQUIRED_ATTRIBUTES if getattr(
            coral, attr, None) is None]
        if missing_attributes:
            self.logger.error(
                f"Game piece {coral} is missing attributes: {', '.join(missing_attributes)}")
            return False
        return True

    def compute_score(self, coral: Coral) -> Optional[float]:
        """Calculate the weighted score for a game piece."""
        if not (coral.relative_angle_deg and coral.confidence and coral.relative_distance_mm):
            return None
        
        return (
            AutoCoralConfig.CORAL_CONFIDENCE_WEIGHT_PCT * coral.confidence +
            AutoCoralConfig.CORAL_DISTANCE_WEIGHT_PCT * ((120 - coral.relative_distance_mm) / 120) +
            AutoCoralConfig.CORAL_ANGULAR_WEIGHT_PCT *
            (1 - abs(coral.relative_angle_deg) / 180)
        )