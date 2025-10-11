import logging
from typing import Any, Dict, List

###################################################################


class DebugConfig:
    TESTING: bool = True
    DEFAULT_TASK: str = "test"
    DEFAULT_KEY: str = "1"
    TASK_KEYS: List = ["1", "2"]


class FieldConfig:
    FIELD_WIDTH_M = 17.55
    FIELD_HEIGHT_M = 8.05


class CameraConfig:     # Logitech C920
    HORIZONTAL_FOV_DEG: float = 59.6
    FOCAL_LENGTH_MM: float = 3.725
    FRAME_WIDTH_PX: int = 640
    FRAME_HEIGHT_PX: int = 640
    DIAGONAL_SENSOR_WIDTH_MM: float = 6
    SPACE_BETWEEN_STEREO_CAMERAS_MM: float = 0.0
    VISION_RANGE_M: float = 4.0


class DisplayConfig:
    """Configuration settings for video output."""
    WINDOW_TITLE: str = 'Output Video'
    SHOW_VIDEO: bool = True
    SAVE_VIDEO: bool = True
    ROTATE_IMAGE: bool = False
    FLIP_IMAGE_HORIZONTALLY: bool = False
    FLIP_IMAGE_VERTICALLY: bool = False
    INPUT_PATH: str | int = "test/input/video.mp4"  # "http://limelight.local:5800" #
    OUTPUT_VIDEO_PATH: str = 'test/output/output.avi'
    APRILTAG_CROSSHAIR_LINE_LENGTH = 10
    LABEL_COLOURS: Dict[str, List[int]] = {
        "0": [85, 186, 151],    # Algae
        "1": [0, 0, 0],         # Cage
        "4": [255, 255, 255],   # Coral
        "5": [255, 0, 0],       # Robot
    }


class DetectorConfig:
    IOU_THRESHOLD: float = 0.5
    CONFIDENCE_THRESHOLD: float = 0.5
    WEIGHTS_LOCATION: str = 'vision/weights/best.onnx'


class AprilTagConfig:
    APRILTAG_SIZE_CM: float = 22.86


class RamFernoRobotConfig:
    ROBOT_WIDTH: float


class SelfDrivingConfig:
    MAX_SELF_DRIVING_SPEED: float = 1.0
    MAX_SELF_DRIVING_ROTATIONAL_RATE: float = 1 / 180.0


class AutoProcessorConfig:
    PROCESSOR_DESIRED_DISTANCE_MM: float = 5.0
    PROCESSOR_MAX_DISTANCE_MM: float = 120.0


class AutoReefConfig:
    REEF_DESIRED_DISTANCE_MM: float = 5.0
    REEF_MAX_DISTANCE_MM: float = 120.0


class AutoAlgaeConfig:
    ALGAE_SIZE_MM: float = 413

    ALGAE_CONFIDENCE_WEIGHT_PCT: float = 1.0
    ALGAE_DISTANCE_WEIGHT_PCT: float = 1.0
    ALGAE_ANGULAR_WEIGHT_PCT: float = 1.0

    ALGAE_DESIRED_DISTANCE_MM = 10.0
    ALGAE_MAX_DISTANCE_MM = 120.0


class AutoCoralConfig:
    CORAL_SIZE_MM: float = 11
    CORAL_DESIRED_DISTANCE_MM: float = 10.0
    CORAL_MAX_DISTANCE_MM: float = 120.0

    CORAL_CONFIDENCE_WEIGHT_PCT: float = 1.0
    CORAL_DISTANCE_WEIGHT_PCT: float = 1.0
    CORAL_ANGULAR_WEIGHT_PCT: float = 1.0


class AutoRobotConfig:
    AVERAGE_ROBOT_SIZE_MM: float = 457


class AutoHangConfig:
    CAGE_WIDTH_MM: float = 190

    POLE_STRAFING_MINIMUM_PCT: float = 0.05
    POLE_STRAFING_MAXIMUM_PCT: float = 0.4

    CAGE_CENTERED_WEIGHT_PCT: float = 0.5
    CAGE_SIZE_WEIGHT_PCT: float = 0.5
    MISSING_CAGE_SPEED_PCT: float = 0.2


class LoggingConfig:
    FPS_LOGGING_RATE: int = 200  # 1/x Frames
    LOG_LEVEL: int = logging.DEBUG


class NetworkingConfig:
    ROBOT_IP_ADDRESS: str = "10.37.56.2"
    NETWORK_TABLE_NAME: str = "AIPipeline"
    DATA_ENTRY_NAME: str = "data"
