import math

from config import CameraConfig

class MonoVision:
    @staticmethod
    def get_distance_to_object_in_mm(object_width_in_mm, object_width_in_pixels) -> float:
        """Calculate the distance and angle offset of an object."""
        return (object_width_in_mm * (CameraConfig.FRAME_WIDTH / (2 * math.tan(math.radians(CameraConfig.HORIZONTAL_FOV) / 2)))) / object_width_in_pixels
    
    @staticmethod
    def get_angle_to_object_in_degrees(object_x) -> float:
        """Calculate the distance and angle offset of an object."""
        return math.degrees(math.atan((object_x - (CameraConfig.FRAME_WIDTH / 2)) / (CameraConfig.FRAME_WIDTH / (2 * math.tan(math.radians(CameraConfig.HORIZONTAL_FOV) / 2)))))