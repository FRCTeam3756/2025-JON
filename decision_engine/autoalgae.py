import os
import logging

from config import AutoAlgaeConfig
from decision_engine.trackable_objects import *

script_name = os.path.splitext(os.path.basename(__file__))[0]

log_file = os.path.join("logs", f"{script_name}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

################################################

class DecisionMatrix:
    REQUIRED_ATTRIBUTES = ['confidence', 'distance', 'angle']
    
    def __init__(self) -> None:
        pass

    def compute_best_game_piece(self, *game_pieces: Object) -> Object:
        """Compute the best game piece based on weighted attributes."""
        if not game_pieces:
            return
        
        best_piece = None
        
        for piece in game_pieces:
            if self.validate_game_piece(piece):
                if best_piece is None or self.compute_score(piece) > self.compute_score(best_piece):
                    best_piece = piece
        
        if best_piece is None:
            logging.info("All pieces suck.")
        return best_piece
        
    def validate_game_piece(self, game_piece: Object) -> Object:
        """Check if a game piece has all required attributes."""
        missing_attributes = [attr for attr in self.REQUIRED_ATTRIBUTES if getattr(game_piece, attr, None) is None]
        if missing_attributes:
            logging.error(f"Game piece {game_piece} is missing attributes: {', '.join(missing_attributes)}")
            return False
        return True
    
    def compute_score(self, game_piece):
        """Calculate the weighted score for a game piece."""
        return (
            AutoAlgaeConfig.ALGAE_CONFIDENCE_WEIGHT * game_piece.confidence +
            AutoAlgaeConfig.ALGAE_DISTANCE_WEIGHT * ((120 - game_piece.distance) / 120) +
            AutoAlgaeConfig.ALGAE_ANGULAR_WEIGHT * (1 - abs(game_piece.angle) / 180)
        )