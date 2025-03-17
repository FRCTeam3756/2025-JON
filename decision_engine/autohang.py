import os
import logging
from typing import List, Tuple
from config import DisplayConfig
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

    def get_autohang_command(self, cages: List[List[float]], poles: List[List[float]], chains: List[List[float]]) -> Tuple[float, float, float, bool]:
        x, y, rot = 0.0, 0.0, 0.0

        cage: List[float] = self.find_best_cage(cages)
        if not cage:
            self.logger.warning("No cage found")
            return [0.0, 0.0, 0.0, False]

        chain: List[float] = self.find_closest_chain(chains, cage)
        poles: List[List[float]] = self.get_valid_poles(cage, poles)
        front_left_pole, front_right_pole = self.get_front_poles(poles)

        y = self.get_strafe_amount(front_left_pole, front_right_pole)
        if chain:
            x = self.get_driving_speed(cage)
            rot = self.get_rotation_amount(chain)
        else:
            x = self.CAGE_NOT_FOUND_SPEED

        self.logger.info(x, y, rot, True)
        return [x, y, rot, True]

    def find_best_cage(self, cages: list[list]) -> list:
        if not cages:
            return []

        best_cage: List[float] = max(
            cages,
            key=lambda cage: ((cage[2] / DisplayConfig.FRAME_WIDTH) * self.CAGE_SIZE_WEIGHT) +
            ((1 - abs(cage[0] - DisplayConfig.FRAME_WIDTH / 2) /
             (DisplayConfig.FRAME_WIDTH / 2)) * self.CAGE_CENTERED_WEIGHT)
        )
        return best_cage

    def find_closest_chain(self, chains: List[List[float]], cage: List[float]) -> List[float]:
        if not chains:
            return []

        cage_left: float = cage[0] - cage[2] / 2
        cage_right: float = cage[0] + cage[2] / 2
        cage_top: float = cage[1] - (cage[2] * cage[3]) / 2

        closest_chain: List[float] = max(
            chains,
            key=lambda chain: 0.5 if (chain[1] + (chain[2] * chain[3]) / 2 > cage_top
                                      or chain[0] + chain[2] / 2 < cage_left
                                      or chain[0] - chain[2] / 2 > cage_right)
            else 0
        )

        return closest_chain

    def get_valid_poles(self, cage: List[float], poles: List[List[float]]) -> List[List[float]]:
        if not poles:
            return []

        x_tol: float = self.clamp(cage[2] * self.POLE_TOLERANCE_PERCENTAGE,
                                  self.POLE_MINIMUM_TOLERANCE, self.POLE_MAXIMUM_TOLERANCE)
        y_tol: float = self.clamp(cage[2] * cage[3] * self.POLE_TOLERANCE_PERCENTAGE,
                                  self.POLE_MINIMUM_TOLERANCE, self.POLE_MAXIMUM_TOLERANCE)

        return [pole for pole in poles if abs(pole[0] - cage[0]) < cage[2] / 2 + x_tol and abs(pole[1] - cage[1]) < cage[3] / 2 + y_tol]

    def get_front_poles(self, poles: List[List[float]]) -> Tuple[List[float], List[float]]:
        if len(poles) < 2:
            return [], []

        sorted_poles = sorted(poles, key=lambda p: p[2], reverse=True)[:2]
        return sorted(sorted_poles, key=lambda p: p[0])

    def get_strafe_amount(self, front_left_pole: List[float], front_right_pole: List[float]) -> float:
        if not front_left_pole and not front_right_pole:
            return 0.0

        pole_size_difference = front_right_pole[2] - front_left_pole[2]
        normalized_difference = pole_size_difference / self.FRAME_WIDTH

        strafe_amount = self.clamp(
            normalized_difference * 2, -self.POLE_STRAFING_MAXIMUM, self.POLE_STRAFING_MINIMUM)

        return 0.0 if abs(strafe_amount) < self.POLE_STRAFING_MINIMUM else strafe_amount

    def get_driving_speed(self, cage: list) -> float:
        return cage[2] / 640 if cage else 0.0

    def get_rotation_amount(self, chain: list[list]) -> float:
        return (chain[0] - self.FRAME_WIDTH / 2) / (self.FRAME_WIDTH / 2) if chain else 0.0
