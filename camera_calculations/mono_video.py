import math
from config import CameraConfig

class MonoVision:
    def __init__(self):
        self.center_x = CameraConfig.FRAME_WIDTH / 2
        self.fov_rad = math.radians(CameraConfig.HORIZONTAL_FOV)
        
        self.focal_length = CameraConfig.FRAME_WIDTH / (2 * math.tan(self.fov_rad / 2))

    def find_distance_and_angle(self, object_x, object_width_in_mm, object_width_in_pixels)-> tuple:
        """Calculate the distance and angle offset of an object."""
        distance = (object_width_in_mm * self.focal_length) / object_width_in_pixels
        angle_offset = math.degrees(math.atan((object_x - self.center_x) / self.focal_length))
        return distance, angle_offset