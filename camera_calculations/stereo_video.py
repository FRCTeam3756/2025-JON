import math

class StereoVision:
    def __init__(self):
        self.inches_between_cameras = 36   # Dinches
        self.frame_width = 640             # in pixels
        self.frame_height = 640            # in pixels
        self.focal_length = 2.75           # in mm
        self.sensor_width = 4              # in mm
        
        self.focal_length_in_pixels = (self.frame_width / self.sensor_width) * self.focal_length

    def calculate_disparity(self, left_camera_box, right_camera_box):
        """Calculate disparity between the left and right camera boxes."""
        x1 = left_camera_box[0]
        x2 = right_camera_box[0]
        return abs(x1 - x2)

    def calculate_distance(self, disparity):
        """Calculate the distance to an object based on disparity."""
        if disparity == 0:
            return float('inf')  # Return infinity if object is too far away
        
        return (self.focal_length_in_pixels * self.inches_between_cameras) / disparity

    def calculate_angle(self, left_camera_box, right_camera_box):
        """Calculate the angle of deviation from the center of the frame."""
        center_x = (left_camera_box[0] + right_camera_box[0]) / 2
        deviation = center_x - (self.frame_width / 2)

        if abs(deviation) < 1e-6:
            return 0.0

        angle_radians = math.atan(deviation / self.focal_length_in_pixels)
        return math.degrees(angle_radians)
    
    def analyze(self, left_camera_box, right_camera_box):
        """Analyze disparity, distance, and angle for the given camera boxes."""
        if not left_camera_box or not right_camera_box:
            return float('inf'), 0.0
        
        disparity = self.calculate_disparity(left_camera_box, right_camera_box)
        distance = self.calculate_distance(disparity)
        angle = self.calculate_angle(left_camera_box, right_camera_box)

        return distance, angle