import math

class MonoVision:
    def __init__(self):
        self.image_width = 640        # in pixels
        self.image_height = 640       # in pixels
        self.fov_horizontal = 59.6    # in degrees
        self.note_width = 12          # in inches

        self.center_x = self.image_width / 2
        self.fov_rad = math.radians(self.fov_horizontal)
        
        self.focal_length = self.image_width / (2 * math.tan(self.fov_rad / 2))

    def find_distance_and_angle(self, object_x, object_width)-> tuple:
        """Calculate the distance and angle offset of an object."""
        distance = (self.note_width * self.focal_length) / object_width
        angle_offset = math.degrees(math.atan((object_x - self.center_x) / self.focal_length))
        return distance, angle_offset