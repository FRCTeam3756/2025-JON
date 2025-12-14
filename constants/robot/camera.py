"""
Copyright (c) FRC Team 3756 RamFerno.
Open Source Software; you can modify and/or share it under the terms of
the license viewable in the root directory of this project.
"""

from dataclasses import dataclass
from enum import Enum


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