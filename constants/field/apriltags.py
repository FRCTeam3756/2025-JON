"""
Copyright (c) FRC Team 3756 RamFerno.
Open Source Software; you can modify and/or share it under the terms of
the license viewable in the root directory of this project.
"""

from typing import Dict, Tuple


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