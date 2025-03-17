from typing import Any, Dict, List

###################################################################

class DebugConfig:
    TESTING: bool = True
    DEFAULT_TASK: str = "auto"

class DisplayConfig:
    """Configuration settings for video output."""
    WINDOW_TITLE: str = 'Output Video'
    SHOW_VIDEO: bool = True
    SAVE_VIDEO: bool = True
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 640
    ROTATE_IMAGE: bool = False
    FLIP_IMAGE_HORIZONTALLY: bool = False
    FLIP_IMAGE_VERTICALLY: bool = False
    INPUT_VIDEO_PATH: Any = "test/input/video3.mp4" #"http://limelight.local:5800" #0 #
    OUTPUT_VIDEO_PATH: str = 'test/output/output.mp4'
    LABEL_COLOURS: Dict[str, List[int]] = {
        "0": [85, 186, 151],    # Algae
        "1": [0, 0, 0],         # Cage
        "2": [0, 0, 255],       # Cage Pole
        "3": [149, 149, 149],   # Chain
        "4": [255, 255, 255],   # Coral
        "5": [121, 217, 255],   # Robot
    }

class YOLOConfig:
    IOU_THRESHOLD: float = 0.4
    CONFIDENCE_THRESHOLD: float = 0.7
    WEIGHTS_LOCATION: str = 'vision_tracking/weights/best.onnx'

class SelfDrivingConfig:
    MAX_SELF_DRIVING_SPEED = 1.0
    MAX_SELF_DRIVING_ROTATIONAL_RATE = 1 / 180.0

class AutoAlgaeConfig:
    ALGAE_CONFIDENCE_WEIGHT: float = 1.0
    ALGAE_DISTANCE_WEIGHT: float = 1.0
    ALGAE_ANGULAR_WEIGHT: float = 1.0

    ALGAE_DESIRED_DISTANCE = 10.0 
    ALGAE_MAX_DISTANCE = 120.0

class AutoHangConfig:
    POLE_TOLERANCE_PERCENTAGE: float = 0.125
    POLE_MINIMUM_TOLERANCE: int = DisplayConfig.FRAME_WIDTH // 100
    POLE_MAXIMUM_TOLERANCE: int = DisplayConfig.FRAME_WIDTH // 10
    POLE_STRAFING_MINIMUM: float = 0.05
    POLE_STRAFING_MAXIMUM: float = 0.4

    CAGE_CENTERED_WEIGHT: float = 0.5
    CAGE_SIZE_WEIGHT: float = 0.5
    CAGE_NOT_FOUND_SPEED: float = 0.2

class LoggingConfig:
    FPS_LOGGING_RATE: int = 200

class NetworkingConfig:
    ROBOT_IP_ADDRESS: str = "10.37.56.2"
    NETWORK_TABLE_NAME: str = "AIPipeline"
    DATA_ENTRY_NAME: str = "data"