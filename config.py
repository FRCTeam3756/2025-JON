import logging
from enum import Enum
from typing import Dict, List, Tuple
from dataclasses import dataclass

###################################################################


class DebugConfig:
    TESTING: bool = True
    TASKS: List[str] = ["pickup_algae", "score_processor"]
    TASK: int = 0

class FieldConfig:
    FIELD_WIDTH_M: float = 17.55
    FIELD_HEIGHT_M: float = 8.05


@dataclass(frozen=True)
class CameraParams:
    HORIZONTAL_FOV_DEG: float
    NATIVE_FRAME_WIDTH_PX: int
    NATIVE_FRAME_HEIGHT_PX: int
    DIAGONAL_SENSOR_WIDTH_MM: float
    HORIZONTAL_SENSOR_WIDTH_MM: float
    SPACE_BETWEEN_STEREO_CAMERAS_MM: float
    VISION_RANGE_M: float
    FOCAL_LENGTH_MM: float

    @property
    def FOCAL_LENGTH_PX(self) -> float:
        return self.FOCAL_LENGTH_MM * self.NATIVE_FRAME_WIDTH_PX / self.HORIZONTAL_SENSOR_WIDTH_MM


class Cameras(Enum):
    INSTA360_X4 = CameraParams(
        HORIZONTAL_FOV_DEG=110,
        NATIVE_FRAME_WIDTH_PX=1280,
        NATIVE_FRAME_HEIGHT_PX=720,
        DIAGONAL_SENSOR_WIDTH_MM=9.06,
        HORIZONTAL_SENSOR_WIDTH_MM=6.4,
        SPACE_BETWEEN_STEREO_CAMERAS_MM=0.0,
        VISION_RANGE_M=20.0,
        FOCAL_LENGTH_MM=6.7
    )
    LOGITECH_C920 = CameraParams(
        HORIZONTAL_FOV_DEG=70.42,
        NATIVE_FRAME_WIDTH_PX=1920,
        NATIVE_FRAME_HEIGHT_PX=1080,
        DIAGONAL_SENSOR_WIDTH_MM=6.0,
        HORIZONTAL_SENSOR_WIDTH_MM=4.8,
        SPACE_BETWEEN_STEREO_CAMERAS_MM=0.0,
        VISION_RANGE_M=20.0,
        FOCAL_LENGTH_MM=3.67
    )
    LIFECAM_HD3000 = CameraParams(
        HORIZONTAL_FOV_DEG=68,
        NATIVE_FRAME_WIDTH_PX=1280,
        NATIVE_FRAME_HEIGHT_PX=720,
        DIAGONAL_SENSOR_WIDTH_MM=4.14,
        HORIZONTAL_SENSOR_WIDTH_MM=3.6,
        SPACE_BETWEEN_STEREO_CAMERAS_MM=0.0,
        VISION_RANGE_M=20.0,
        FOCAL_LENGTH_MM=4.2
    )

CameraConfig = Cameras.INSTA360_X4.value

class DisplayConfig:
    """Configuration settings for video output."""
    FRAME_WIDTH_PX: int = 640
    FRAME_HEIGHT_PX: int = 640
    WINDOW_TITLE: str = 'Output Video'
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
    WEIGHTS_LOCATION: str = 'src/vision/weights/'


class AprilTagConfig:
    TAG_FAMILY: str = "tag36h11"
    APRILTAG_SIZE_MM: float = 165.1  # Inner Square

    _APRILTAG_POSITIONS_INCHES: Dict[int, Tuple[float, float, int]] = {
        1: (656.98, 24.73, 126),    # (X: Inches, Y: Inches, ROT: Degrees)
        2: (656.98, 291.90, 234),
        3: (452.40, 316.21, 270),
        4: (365.20, 241.44, 0),
        5: (365.20, 75.19, 0),
        6: (530.49, 129.97, 300),
        7: (546.87, 158.30, 0),
        8: (530.49, 186.63, 60),
        9: (497.77, 186.63, 120),
        10: (481.39, 158.30, 180),
        11: (497.77, 129.97, 240),
        12: (33.91, 24.73, 54),
        13: (33.91, 291.90, 306),
        14: (325.68, 241.44, 180),
        15: (325.68, 75.19, 180),
        16: (238.49, 0.42, 90),
        17: (160.39, 129.97, 240),
        18: (144.00, 158.30, 180),
        19: (160.39, 186.63, 120),
        20: (193.10, 186.63, 60),
        21: (209.49, 158.30, 0),
        22: (193.10, 129.97, 300),
    }
    APRILTAG_POSITIONS_M: Dict[int, Tuple[float, float, int]] = {
        tag_id: (x_in * 0.0254, y_in * 0.0254, rot_deg)
        for tag_id, (x_in, y_in, rot_deg) in _APRILTAG_POSITIONS_INCHES.items()
    }


class RamFernoRobotConfig:
    ROBOT_WIDTH_M: float = 0.8128   # 32" with Bumpers
    ROBOT_LENGTH_M: float = 0.8128  # 32" with Bumpers


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
