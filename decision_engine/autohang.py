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

class HangDriveCommand:
    FRAME_WIDTH, FRAME_HEIGHT = 640, 640

    POLE_TOLERANCE_PERCENTAGE = 0.125
    POLE_MINIMUM_TOLERANCE = FRAME_WIDTH // 100
    POLE_MAXIMUM_TOLERANCE = FRAME_WIDTH // 10
    POLE_STRAFING_MINIMUM = 0.05
    POLE_STRAFING_MAXIMUM = 0.4

    CAGE_CENTERED_WEIGHT = 0.5
    CAGE_SIZE_WEIGHT = 0.5
    CAGE_NOT_FOUND_SPEED = 0.2

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

    @staticmethod
    def clamp(input: float, minimum: float, maximum: float) -> float:
        return max(min(input, maximum), minimum)

    def return_robot_command(self, cages: list[list], poles: list[list], chains: list[list]) -> list[float, float, float, int]:
        x, y, rot = 0.0, 0.0, 0.0

        cage = self.find_best_cage(cages)
        if not cage:
            return [0.0, 0.0, 0.0, 1]

        chain = self.find_closest_chain(chains, cage)
        poles = self.get_valid_poles(cage, poles)
        front_left_pole, front_right_pole = self.get_front_poles(poles)

        y = self.get_strafe_amount(front_left_pole, front_right_pole)
        if chain:
            x = self.get_driving_speed(cage)
            rot = self.get_rotation_amount(chain)
        else:
            x = self.CAGE_NOT_FOUND_SPEED

        return [x, y, rot, 0]

    def find_best_cage(self, cages: list[list]) -> list:
        if not cages:
            return []

        best_cage = max(
            cages,
            key=lambda cage: ((cage[2] / self.FRAME_WIDTH) * self.CAGE_SIZE_WEIGHT) +
            ((1 - abs(cage[0] - self.FRAME_WIDTH / 2) /
             (self.FRAME_WIDTH / 2)) * self.CAGE_CENTERED_WEIGHT)
        )
        return best_cage

    def find_closest_chain(self, chains: list[list], cage) -> list:
        if not chains:
            return []

        cage_left, cage_right = cage[0] - cage[2] / 2, cage[0] + cage[2] / 2
        cage_top = cage[1] - (cage[2] * cage[3]) / 2

        closest_chain = max(
            chains,
            key=lambda chain: 0.5 if (chain[1] + (chain[2] * chain[3]) / 2 > cage_top
                                      or chain[0] + chain[2] / 2 < cage_left
                                      or chain[0] - chain[2] / 2 > cage_right)
            else 0
        )

        return closest_chain

    def get_valid_poles(self, cage, poles) -> list[list]:
        if not poles:
            return []

        x_tol = self.clamp(cage[2] * self.POLE_TOLERANCE_PERCENTAGE, self.POLE_MINIMUM_TOLERANCE, self.POLE_MAXIMUM_TOLERANCE)
        y_tol = self.clamp(cage[2] * cage[3] * self.POLE_TOLERANCE_PERCENTAGE, self.POLE_MINIMUM_TOLERANCE, self.POLE_MAXIMUM_TOLERANCE)
        
        return [pole for pole in poles if abs(pole[0] - cage[0]) < cage[2] / 2 + x_tol and abs(pole[1] - cage[1]) < cage[3] / 2 + y_tol]

    def get_front_poles(self, poles: list[list]) -> float:
        if len(poles) < 2:
            return [], []

        sorted_poles = sorted(poles, key=lambda p: p[2], reverse=True)[:2]
        return sorted(sorted_poles, key=lambda p: p[0])

    def get_strafe_amount(self, front_left_pole: list, front_right_pole: list) -> float:
        if not front_left_pole and not front_right_pole:
            return 0.0

        pole_size_difference = front_right_pole[2] - front_left_pole[2]
        normalized_difference = pole_size_difference / self.FRAME_WIDTH

        strafe_amount = self.clamp(normalized_difference * 2, -self.POLE_STRAFING_MAXIMUM, self.POLE_STRAFING_MINIMUM)

        if abs(strafe_amount) < self.POLE_STRAFING_MINIMUM:
            return 0.0

        return strafe_amount

    def get_driving_speed(self, cage: list) -> float:
        if not cage:
            return 0.0

        return cage[2] / 640

    def get_rotation_amount(self, chain: list[list]) -> float:
        if not chain:
            return 0.0

        return (chain[0] - self.FRAME_WIDTH / 2) / (self.FRAME_WIDTH / 2)


if __name__ == '__main__':
    chains = [[400, 450, 10, 7], [180, 250, 10, 7]]
    cages = [[400, 200, 80, 1.5], [180, 50, 70, 1.8]]
    poles = [[380, 190, 20, 2], [410, 210, 20, 2], [370, 200, 20, 2],
             [380, 170, 20, 2], [180, 2, 0, 7], [30, 20, 1, 2]]

    hang_drive_command = HangDriveCommand()
    logging.info(hang_drive_command.return_robot_command(cages, poles, chains))
