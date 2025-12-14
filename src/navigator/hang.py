"""
Copyright (c) FRC Team 3756 RamFerno.
Open Source Software; you can modify and/or share it under the terms of
the license viewable in the root directory of this project.
"""

import os
import logging
from typing import List, Optional, Tuple
from constants.monitoring.display import DisplayConfig
from constants.field.cage import CageConfig
from logs.logging_setup import setup_logger


class HangDriveCommand:
    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        setup_logger(file_name)
        self.logger = logging.getLogger(file_name)

    def clamp(self, input: float, minimum: float, maximum: float) -> float:
        return max(min(input, maximum), minimum)

    def get_autohang_command(self, cages: List[List[float]]) -> Tuple[float, float, float, bool]:
        x, y, rot = 0.0, 0.0, 0.0

        cage = self.find_best_cage(cages)
        if not cage:
            self.logger.warning("No cage found")
            return (0.0, 0.0, 0.0, False)

        y = self.get_strafe_amount(cage)
        if cage:
            x = self.get_driving_speed(cage)
            rot = self.get_rotation_amount(cage)
        else:
            x = CageConfig.MISSING_CAGE_SPEED_PCT

        self.logger.info(x, y, rot, True)
        return (x, y, rot, True)

    def find_best_cage(self, cages: List[List[float]]) -> Optional[List[float]]:
        if not cages:
            return None

        best_cage: List[float] = max(
            cages,
            key=lambda cage: ((cage[2] / DisplayConfig.FRAME_WIDTH_PX) * CageConfig.CAGE_SIZE_WEIGHT_PCT) +
            ((1 - abs(cage[0] - DisplayConfig.FRAME_WIDTH_PX / 2) /
             (DisplayConfig.FRAME_WIDTH_PX / 2)) * CageConfig.CAGE_CENTERED_WEIGHT_PCT)
        )
        return best_cage

    def get_strafe_amount(self, cage: List[float]) -> float:
        if not cage:
            return 0.0

        strafe_amount = (cage[0] - DisplayConfig.FRAME_WIDTH_PX / 2) / \
            (DisplayConfig.FRAME_WIDTH_PX / 2) if cage else 0.0

        strafe_amount = self.clamp(
            strafe_amount, -CageConfig.POLE_STRAFING_MAXIMUM_PCT, CageConfig.POLE_STRAFING_MINIMUM_PCT)

        return 0.0 if abs(strafe_amount) < CageConfig.POLE_STRAFING_MINIMUM_PCT else strafe_amount

    def get_driving_speed(self, cage: List[float]) -> float:
        return cage[2] / 640 if cage else 0.0

    def get_rotation_amount(self, cage: List[float]) -> float:
        return (cage[0] - DisplayConfig.FRAME_WIDTH_PX / 2) / (DisplayConfig.FRAME_WIDTH_PX / 2) if cage else 0.0
