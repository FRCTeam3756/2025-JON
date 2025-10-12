import math

from config import CameraConfig


class MonoVision:
    @staticmethod
    def get_distance_to_object_in_mm(object_width_mm: float, object_width_px: float) -> float:
        """Calculate the distance and angle offset of an object."""
        return (object_width_mm * (CameraConfig.FRAME_WIDTH_PX / (2 * math.tan(math.radians(CameraConfig.HORIZONTAL_FOV_DEG) / 2)))) / object_width_px

    @staticmethod
    def get_angle_to_object_in_degrees(object_x: float) -> float:
        """Calculate the distance and angle offset of an object."""
        return math.degrees(math.atan((object_x - (CameraConfig.FRAME_WIDTH_PX / 2)) / (CameraConfig.FRAME_WIDTH_PX / (2 * math.tan(math.radians(CameraConfig.HORIZONTAL_FOV_DEG) / 2)))))