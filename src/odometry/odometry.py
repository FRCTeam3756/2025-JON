"""
Copyright (c) FRC Team 3756 RamFerno.
Open Source Software; you can modify and/or share it under the terms of
the license viewable in the root directory of this project.
"""

import os
from typing import Optional, Tuple

from logs.logging_setup import setup_logger

###############################################################

class Odometry:
    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        self.logger = setup_logger(file_name)
        self.logger.info("Odometry logger initialized.")

    def get_world_position(self, last_position: Tuple[float, float, float]) -> Optional[Tuple[float, float, float]]:
        # Query to the robot for encoder info
        # Use that to figure out the most likely position
        return None