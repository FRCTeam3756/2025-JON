"""
Copyright (c) FRC Team 3756 RamFerno.
Open Source Software; you can modify and/or share it under the terms of
the license viewable in the root directory of this project.
"""


class DetectorConfig:
    IOU_THRESHOLD: float = 0.5
    CONFIDENCE_THRESHOLD: float = 0.5
    WEIGHTS_LOCATION: str = 'src/vision/weights/'