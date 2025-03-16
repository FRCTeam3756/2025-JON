import os
import logging
from logs.logging_setup import setup_logger

from config import AutoAlgaeConfig
from decision_engine.trackable_objects import *

################################################


class DecisionMatrix:
    REQUIRED_ATTRIBUTES = ['confidence', 'distance', 'angle']

    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        setup_logger(file_name)
        self.logger = logging.getLogger(file_name)

    def compute_best_game_piece(self, *game_pieces: Object) -> Object:
        """Compute the best game piece based on weighted attributes."""
        if not game_pieces:
            return

        best_piece = None

        for piece in game_pieces:
            if self.validate_game_piece(piece):
                if best_piece is None or self.compute_score(piece) > self.compute_score(best_piece):
                    best_piece = piece

        return best_piece

    def validate_game_piece(self, game_piece: Object) -> Object:
        """Check if a game piece has all required attributes."""
        missing_attributes = [attr for attr in self.REQUIRED_ATTRIBUTES if getattr(
            game_piece, attr, None) is None]
        if missing_attributes:
            self.logger.error(
                f"Game piece {game_piece} is missing attributes: {', '.join(missing_attributes)}")
            return False
        return True

    def compute_score(self, game_piece):
        """Calculate the weighted score for a game piece."""
        return (
            AutoAlgaeConfig.ALGAE_CONFIDENCE_WEIGHT * game_piece.confidence +
            AutoAlgaeConfig.ALGAE_DISTANCE_WEIGHT * ((120 - game_piece.distance) / 120) +
            AutoAlgaeConfig.ALGAE_ANGULAR_WEIGHT *
            (1 - abs(game_piece.angle) / 180)
        )
