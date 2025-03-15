import os
import json
import logging

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
        self.get_weighting()
        
    def get_weighting(self):
        try:
            with open("decision_engine/weights.json", "r") as config_file:
                config = json.load(config_file)
                self.confidence_weight = config.get("confidence_weight", 1.0)
                self.distance_weight = config.get("distance_weight", 1.0)
                self.angle_weight = config.get("angle_weight", 1.0)  
        except FileNotFoundError:
            logging.error("Error: 'weights.json' file not found.")
            raise
        except json.JSONDecodeError:
            logging.error("Error: 'weights.json' contains invalid JSON.")
            raise
        except Exception as e:
            logging.error(f"An unexpected error occurred: {e}")
            raise

    def compute_best_game_piece(self, *game_pieces):
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
        
    def validate_game_piece(self, game_piece):
        """Check if a game piece has all required attributes."""
        missing_attributes = [attr for attr in self.REQUIRED_ATTRIBUTES if getattr(game_piece, attr, None) is None]
        if missing_attributes:
            logging.error(f"Game piece {game_piece} is missing attributes: {', '.join(missing_attributes)}")
            return False
        return True
    
    def compute_score(self, game_piece):
        """Calculate the weighted score for a game piece."""
        return (
            self.confidence_weight * game_piece.confidence +
            self.distance_weight * ((120 - game_piece.distance) / 120) +
            self.angle_weight * (1 - abs(game_piece.angle) / 180)
        )