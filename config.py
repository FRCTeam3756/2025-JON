import logging
from enum import Enum
from typing import Dict, List
from dataclasses import dataclass

###################################################################


class DebugConfig:
    TESTING: bool = True
    DEFAULT_TASK: str = "test"
    DEFAULT_KEY: str = "1"
    TASK_KEYS: List = ["1", "2"]


class FieldConfig:
    FIELD_WIDTH_M = 17.55
    FIELD_HEIGHT_M = 8.05


@dataclass(frozen=True)
class CameraParams:
    HORIZONTAL_FOV_DEG: float
    FRAME_WIDTH_PX: int
    FRAME_HEIGHT_PX: int
    DIAGONAL_SENSOR_WIDTH_MM: float
    HORIZONTAL_SENSOR_WIDTH_MM: float
    SPACE_BETWEEN_STEREO_CAMERAS_MM: float
    VISION_RANGE_M: float
    FOCAL_LENGTH_MM: float

    @property
    def FOCAL_LENGTH_PX(self) -> float:
        return self.FOCAL_LENGTH_MM * self.FRAME_WIDTH_PX / self.HORIZONTAL_SENSOR_WIDTH_MM


class Cameras(Enum):
    INSTA360_X4 = CameraParams(
        HORIZONTAL_FOV_DEG=110,
        FRAME_WIDTH_PX=640,
        FRAME_HEIGHT_PX=640,
        DIAGONAL_SENSOR_WIDTH_MM=9.06,
        HORIZONTAL_SENSOR_WIDTH_MM=6.4,
        SPACE_BETWEEN_STEREO_CAMERAS_MM=0.0,
        VISION_RANGE_M=20.0,
        FOCAL_LENGTH_MM=6.7
    )
    LOGITECH_C920 = CameraParams(
        HORIZONTAL_FOV_DEG=70.42,
        FRAME_WIDTH_PX=640,
        FRAME_HEIGHT_PX=640,
        DIAGONAL_SENSOR_WIDTH_MM=6.0,
        HORIZONTAL_SENSOR_WIDTH_MM=4.8,
        SPACE_BETWEEN_STEREO_CAMERAS_MM=0.0,
        VISION_RANGE_M=20.0,
        FOCAL_LENGTH_MM=3.67
    )
    LIFECAM_HD3000 = CameraParams(
        HORIZONTAL_FOV_DEG=68,
        FRAME_WIDTH_PX=640,
        FRAME_HEIGHT_PX=640,
        DIAGONAL_SENSOR_WIDTH_MM=4.14,
        HORIZONTAL_SENSOR_WIDTH_MM=3.6,
        SPACE_BETWEEN_STEREO_CAMERAS_MM=0.0,
        VISION_RANGE_M=20.0,
        FOCAL_LENGTH_MM=4.2
    )

CameraConfig = Cameras.INSTA360_X4.value

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
    TAG_FAMILY: str = "tag36h11"
    APRILTAG_SIZE_MM: float = 165.1  # Inner Square


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
