import cv2
import math
import numpy as np
from typing import List, Tuple
from config import DisplayConfig

class VideoDisplay:
    @staticmethod
    def show_frame(window_name: str, frame: np.ndarray) -> None:
        """Displays the frame in a window."""
        cv2.imshow(window_name, frame)

    @staticmethod
    def annotate_frame(frame: np.ndarray, boxes: List[Tuple[int, int, int, int]], class_ids: List[int], apriltags) -> np.ndarray:
        """Annotate the frame with bounding boxes and labels."""
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            color = DisplayConfig.LABEL_COLOURS.get(str(class_ids[i]), (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        for apriltag in apriltags:
            frame = VideoDisplay.draw_apriltag(frame, apriltag)

        return frame
    
    @staticmethod
    def insert_text_onto_frame(frame: np.ndarray, messages: List[str]) -> np.ndarray:
        """Annotate the frane with text"""
        for i, message in enumerate(messages):
            cv2.putText(frame, message, (10, (30 + (i * 50))), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 2)

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
    def draw_apriltag(frame, detection):
        """Draws the tag's bounding box, center, and ID on the frame."""
        for i in range(4):
            j = (i + 1) % 4
            point1 = (int(detection.getCorner(i).x), int(detection.getCorner(i).y))
            point2 = (int(detection.getCorner(j).x), int(detection.getCorner(j).y))
            cv2.line(frame, point1, point2, (0, 255, 0), 2)

        center_x = int(detection.getCenter().x)
        center_y = int(detection.getCenter().y)

        cv2.line(frame, (center_x - DisplayConfig.APRILTAG_CROSSHAIR_LINE_LENGTH, center_y), (center_x + DisplayConfig.APRILTAG_CROSSHAIR_LINE_LENGTH, center_y), (0, 0, 255), 2)
        cv2.line(frame, (center_x, center_y - DisplayConfig.APRILTAG_CROSSHAIR_LINE_LENGTH), (center_x, center_y + DisplayConfig.APRILTAG_CROSSHAIR_LINE_LENGTH), (0, 0, 255), 2)

        cv2.putText(frame, str(detection.getId()), (center_x + DisplayConfig.APRILTAG_CROSSHAIR_LINE_LENGTH, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        return frame