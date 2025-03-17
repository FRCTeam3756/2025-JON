import os
import logging
from typing import List, Tuple
from config import DisplayConfig, AutoHangConfig
from logs.logging_setup import setup_logger

class HangDriveCommand:
    REQUIRED_ATTRIBUTES: List[str] = ['confidence', 'distance', 'angle']

    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        setup_logger(file_name)
        self.logger = logging.getLogger(file_name)

    @staticmethod
    def clamp(input: float, minimum: float, maximum: float) -> float:
        return max(min(input, maximum), minimum)

    def get_autohang_command(self, cages: List[List[float]]) -> Tuple[float, float, float, bool]:
        x, y, rot = 0.0, 0.0, 0.0

        cage: List[float] = self.find_best_cage(cages)
        if not cage:
            self.logger.warning("No cage found")
            return [0.0, 0.0, 0.0, False]


        y = self.get_strafe_amount(cage)
        if cage:
            x = self.get_driving_speed(cage)
            rot = self.get_rotation_amount(cage)
        else:
            x = self.CAGE_NOT_FOUND_SPEED

        self.logger.info(x, y, rot, True)
        return [x, y, rot, True]

    def find_best_cage(self, cages: list[list]) -> list:
        if not cages:
            return []

        best_cage: List[float] = max(
            cages,
            key=lambda cage: ((cage[2] / DisplayConfig.FRAME_WIDTH) * AutoHangConfig.CAGE_SIZE_WEIGHT) +
            ((1 - abs(cage[0] - DisplayConfig.FRAME_WIDTH / 2) /
             (DisplayConfig.FRAME_WIDTH / 2)) * AutoHangConfig.CAGE_CENTERED_WEIGHT)
        )
        return best_cage

    def get_strafe_amount(self, cage: List[float]) -> float:
        if not cage:
            return 0.0
        
        strafe_amount = (cage[0] - DisplayConfig.FRAME_WIDTH / 2) / (DisplayConfig.FRAME_WIDTH / 2) if cage else 0.0

        strafe_amount = self.clamp(
            strafe_amount, -AutoHangConfig.POLE_STRAFING_MAXIMUM, AutoHangConfig.POLE_STRAFING_MINIMUM)

        return 0.0 if abs(strafe_amount) < AutoHangConfig.POLE_STRAFING_MINIMUM else strafe_amount

    def get_driving_speed(self, cage: list) -> float:
        return cage[2] / 640 if cage else 0.0

    def get_rotation_amount(self, cage: list[list]) -> float:
        return (cage[0] - DisplayConfig.FRAME_WIDTH / 2) / (DisplayConfig.FRAME_WIDTH / 2) if cage else 0.0
