"""
Copyright (c) FRC Team 3756 RamFerno.
Open Source Software; you can modify and/or share it under the terms of
the license viewable in the root directory of this project.
"""

import cv2
import math
import numpy as np
from typing import List, Tuple, Union

from src.apriltags.apriltags import AprilTagDetection
from constants.monitoring.display import DisplayConfig

################################################

class Display:
    @staticmethod
    def show_frame(window_name: str, frame: np.ndarray) -> None:
        """Displays the frame in a window."""
        cv2.imshow(window_name, frame)

    @staticmethod
    def annotate_frame(frame: np.ndarray, boxes: Union[np.ndarray, List[Tuple[int, int, int, int]]], class_ids: Union[np.ndarray, List[int]], apriltags: List[AprilTagDetection]) -> np.ndarray:
        """Annotate the frame with bounding boxes and labels."""
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            color = DisplayConfig.LABEL_COLOURS.get(str(class_ids[i]), (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        for apriltag in apriltags:
            Display.draw_apriltag(frame, apriltag)

        return frame
    
    @staticmethod
    def insert_text_onto_frame(frame: np.ndarray, messages: List[str]) -> np.ndarray:
        """Annotate the frane with text"""
        for i, message in enumerate(messages):
            cv2.putText(frame, message, (10, (30 + (i * 50))), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

        return frame

    @staticmethod
    def draw_angle_line(frame: np.ndarray, angle: float) -> None:
        """Draws a line at a given angle from the bottom center of the screen."""
        height, width = frame.shape[:2]
        start_point = (width // 2, height - 1)
        length = 100
        end_x = int(start_point[0] + (length * math.sin(math.radians(angle))))
        end_y = int(start_point[1] - (length * math.cos(math.radians(angle))))
        
        cv2.line(frame, start_point, (end_x, end_y), (0, 155, 255), 2)

    @staticmethod
    def draw_apriltag(frame: np.ndarray, apriltag: AprilTagDetection) -> None:
        pts = np.array(apriltag.corners, dtype=np.int32)
        for i in range(4):
            cv2.line(frame, tuple(pts[i]), tuple(pts[(i + 1) % 4]), (0, 255, 0), 2)
        c = tuple(np.array(apriltag.center, dtype=np.int32))
        cv2.drawMarker(frame, c, (0, 0, 255), cv2.MARKER_CROSS, 30, 2)
        text = f"ID {apriltag.tag_id} | {apriltag.relative_distance_m:.2f} m"
        cv2.putText(frame, text, (c[0] + 10, c[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)