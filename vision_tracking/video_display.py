import cv2
import math
import numpy as np
from typing import List, Dict, Tuple

class VideoDisplay:
    @staticmethod
    def show_frame(window_name: str, frame: np.ndarray) -> None:
        """Displays the frame in a window."""
        cv2.imshow(window_name, frame)

    @staticmethod
    def annotate_frame(frame: np.ndarray, boxes: List[Tuple[int, int, int, int]], class_ids: List[int], label_colours: Dict[str, Tuple[int, int, int]]) -> np.ndarray:
        """Annotate the frame with bounding boxes and labels."""
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            color = label_colours.get(str(class_ids[i]), (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        return frame

    @staticmethod
    def draw_angle_line(frame: np.ndarray, angle: float) -> None:
        """Draws a line at a given angle from the bottom center of the screen."""
        height, width = frame.shape[:2]
        start_point = (width // 2, height - 1)
        length = 100
        end_x = int(start_point[0] + length * math.sin(math.radians(angle)))
        end_y = int(start_point[1] - length * math.cos(math.radians(angle)))
        
        cv2.line(frame, start_point, (end_x, end_y), (0, 155, 255), 2)