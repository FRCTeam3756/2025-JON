import math
from typing import List, Tuple

from config import CameraConfig


class StereoVision:
    @staticmethod
    def calculate_disparity(left_camera_box: List[int], right_camera_box: List[int]):
        """Calculate disparity between the left and right camera boxes."""
        x1 = left_camera_box[0]
        x2 = right_camera_box[0]
        return abs(x1 - x2)

    @staticmethod
    def calculate_distance(disparity: float) -> float:
        """Calculate the distance to an object based on disparity."""
        if disparity == 0.0:
            return float('inf')  # Return infinity if object is too far away

        distance = (((CameraConfig.FRAME_WIDTH_PX / CameraConfig.DIAGONAL_SENSOR_WIDTH_MM) *
                    CameraConfig.FOCAL_LENGTH_MM) * CameraConfig.SPACE_BETWEEN_STEREO_CAMERAS_MM) / disparity

        return distance

    @staticmethod
    def calculate_angle_deg(left_camera_box: List[int], right_camera_box: List[int]) -> float:
        """Calculate the angle of deviation from the center of the frame."""
        center_x = (left_camera_box[0] + right_camera_box[0]) / 2
        deviation = center_x - (CameraConfig.FRAME_WIDTH_PX / 2)

        if abs(deviation) < 1e-6:
            return 0.0

        angle_rad = math.atan(deviation / ((CameraConfig.FRAME_WIDTH_PX /
                                            CameraConfig.DIAGONAL_SENSOR_WIDTH_MM) * CameraConfig.FOCAL_LENGTH_MM))
        return math.degrees(angle_rad)

    @staticmethod
    def get_distance_and_angle_to_an_object(left_camera_box: List[int], right_camera_box: List[int]) -> Tuple[float, float]:
        """Analyze disparity, distance, and angle for the given camera boxes."""
        if not left_camera_box or not right_camera_box:
            return float('inf'), 0.0

        disparity = StereoVision.calculate_disparity(
            left_camera_box, right_camera_box)
        distance = StereoVision.calculate_distance(disparity)
        angle = StereoVision.calculate_angle_deg(
            left_camera_box, right_camera_box)

        return distance, angle
