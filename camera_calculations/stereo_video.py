import math
from config import CameraConfig

class StereoVision:
    @staticmethod
    def calculate_disparity(left_camera_box, right_camera_box):
        """Calculate disparity between the left and right camera boxes."""
        x1 = left_camera_box[0]
        x2 = right_camera_box[0]
        return abs(x1 - x2)

    @staticmethod
    def calculate_distance(disparity):
        """Calculate the distance to an object based on disparity."""
        if disparity == 0:
            return float('inf')  # Return infinity if object is too far away
        
        return (((CameraConfig.FRAME_WIDTH / CameraConfig.DIAGONAL_SENSOR_WIDTH) * CameraConfig.FOCAL_LENGTH) * CameraConfig.INCHES_BETWEEN_STEREO_CAMERAS) / disparity

    @staticmethod
    def calculate_angle(left_camera_box, right_camera_box):
        """Calculate the angle of deviation from the center of the frame."""
        center_x = (left_camera_box[0] + right_camera_box[0]) / 2
        deviation = center_x - (CameraConfig.FRAME_WIDTH / 2)

        if abs(deviation) < 1e-6:
            return 0.0

        angle_radians = math.atan(deviation / ((CameraConfig.FRAME_WIDTH / CameraConfig.DIAGONAL_SENSOR_WIDTH) * CameraConfig.FOCAL_LENGTH))
        return math.degrees(angle_radians)
    
    @staticmethod
    def get_distance_and_angle_to_an_object(left_camera_box, right_camera_box):
        """Analyze disparity, distance, and angle for the given camera boxes."""
        if not left_camera_box or not right_camera_box:
            return float('inf'), 0.0
        
        disparity = StereoVision.calculate_disparity(left_camera_box, right_camera_box)
        distance = StereoVision.calculate_distance(disparity)
        angle = StereoVision.calculate_angle(left_camera_box, right_camera_box)

        return distance, angle