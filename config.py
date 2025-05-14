import logging
from typing import Any, Dict, List

###################################################################

class DebugConfig:
    TESTING: bool = True
    DEFAULT_TASK: str = "test"
    DEFAULT_KEY: str = "1"
    TASK_KEYS: List = ["1", "2"]

class CameraConfig:     # Logitech C920
    HORIZONTAL_FOV: float = 59.6    # in degrees
    FOCAL_LENGTH: float = 3.725     # in mm
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 640
    DIAGONAL_SENSOR_WIDTH: float = 6              # in mm
    INCHES_BETWEEN_STEREO_CAMERAS: float = 0.0   # in inches

class DisplayConfig:
    """Configuration settings for video output."""
    WINDOW_TITLE: str = 'Output Video'
    SHOW_VIDEO: bool = True
    SAVE_VIDEO: bool = True
    ROTATE_IMAGE: bool = False
    FLIP_IMAGE_HORIZONTALLY: bool = False
    FLIP_IMAGE_VERTICALLY: bool = False
    INPUT_VIDEO_PATH: Any = "video.mp4" #"test/input/video3.mp4" #"http://limelight.local:5800" #
    OUTPUT_VIDEO_PATH: str = 'test/output/output.mp4'
    APRILTAG_CROSSHAIR_LINE_LENGTH = 10
    LABEL_COLOURS: Dict[str, List[int]] = {
        "0": [85, 186, 151],    # Algae
        "1": [0, 0, 0],         # Cage
        "4": [255, 255, 255],   # Coral
        "5": [255, 0, 0],       # Robot
    }

class YOLOConfig:
    IOU_THRESHOLD: float = 0.4
    CONFIDENCE_THRESHOLD: float = 0.7
    WEIGHTS_LOCATION: str = 'vision_tracking/weights/best.engine'

class AprilTagConfig:
    APRILTAG_SIZE_IN_INCHES = 9
    APRILTAG_SIZE_IN_CM = APRILTAG_SIZE_IN_INCHES * 2.54

class SelfDrivingConfig:
    MAX_SELF_DRIVING_SPEED = 1.0
    MAX_SELF_DRIVING_ROTATIONAL_RATE = 1 / 180.0

class AutoProcessorConfig:
    PROCESSOR_DESIRED_DISTANCE_IN_MM = 5.0 
    PROCESSOR_MAX_DISTANCE_IN_MM = 120.0

class AutoAlgaeConfig:
    ALGAE_SIZE_IN_MM: float = 413
    
    ALGAE_CONFIDENCE_WEIGHT: float = 1.0
    ALGAE_DISTANCE_WEIGHT: float = 1.0
    ALGAE_ANGULAR_WEIGHT: float = 1.0

    ALGAE_DESIRED_DISTANCE_IN_MM = 10.0 
    ALGAE_MAX_DISTANCE_IN_MM = 120.0

class AutoCoralConfig:
    CORAL_SIZE_IN_MM: float = 11

class AutoRobotConfig:
    AVERAGE_ROBOT_SIZE_IN_MM: float = 711

class AutoHangConfig:
    CAGE_WIDTH_IN_MM: float = 190

    POLE_TOLERANCE_PERCENTAGE: float = 0.125
    POLE_MINIMUM_TOLERANCE: int = CameraConfig.FRAME_WIDTH // 100
    POLE_MAXIMUM_TOLERANCE: int = CameraConfig.FRAME_WIDTH // 10
    POLE_STRAFING_MINIMUM: float = 0.05
    POLE_STRAFING_MAXIMUM: float = 0.4

    CAGE_CENTERED_WEIGHT: float = 0.5
    CAGE_SIZE_WEIGHT: float = 0.5
    CAGE_NOT_FOUND_SPEED: float = 0.2

class LoggingConfig:
    FPS_LOGGING_RATE: int = 200
    LOG_LEVEL: int = logging.DEBUG

class NetworkingConfig:
    ROBOT_IP_ADDRESS: str = "10.37.56.2"
    NETWORK_TABLE_NAME: str = "AIPipeline"
    DATA_ENTRY_NAME: str = "data"