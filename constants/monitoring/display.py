"""
Copyright (c) FRC Team 3756 RamFerno.
Open Source Software; you can modify and/or share it under the terms of
the license viewable in the root directory of this project.
"""

from typing import Dict, List

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