import os
import math
import logging
import numpy as np
from logs.logging_setup import setup_logger
from typing import List, Optional

import cv2
import robotpy_apriltag as apriltag
from robotpy_apriltag import AprilTagDetection, AprilTagDetector

from config import CameraConfig, AprilTagConfig, DisplayConfig


class AprilTagFinder:
    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        setup_logger(file_name)
        self.logger = logging.getLogger(file_name)

        self.apriltag_detector = AprilTagDetector()
        self.apriltag_detector.addFamily("tag36h11", 3)

    @staticmethod
    def estimate_distance(apriltag: AprilTagDetection) -> float:
        """Estimate distance to the tag based on its size in the image."""
        apriltag_width_in_pixels = apriltag.getCorner(
            0).x - apriltag.getCorner(3).x
        return (CameraConfig.FOCAL_LENGTH_MM * AprilTagConfig.APRILTAG_SIZE_CM) / apriltag_width_in_pixels

    @staticmethod
    def calculate_anglular_diviation(apriltag: AprilTagDetection) -> float:
        """Calculate the angle offset of an apriltag."""
        return math.degrees(math.atan((apriltag.getCenter().x - (CameraConfig.FRAME_WIDTH_PX / 2)) / CameraConfig.FOCAL_LENGTH_MM))

    def find_apriltags(self, frame: np.ndarray) -> List[AprilTagDetection]:
        """Main loop for processing frames and sending drive instructions."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        apriltags = self.apriltag_detector.detect(gray_frame)
        return apriltags
    
    @staticmethod
    def get_best_apriltag(apriltags: List[AprilTagDetection]) -> Optional[AprilTagDetection]:
        """Select the best AprilTag"""
        if not apriltags:
            return None

        cx = CameraConfig.FRAME_WIDTH_PX / 2
        cy = CameraConfig.FRAME_HEIGHT_PX / 2

        def score(tag: AprilTagDetection) -> float:
            center = tag.getCenter()

            points = [tag.getCorner(i) for i in range(4)]

            size = sum(
                math.hypot(points[i].x - points[(i + 1) % 4].x, points[i].y - points[(i + 1) % 4].y)
                for i in range(4)
            ) / 4

            offset = math.hypot(center.x - cx, center.y - cy)
            return (size / 5) - (offset / 10)

        return max(apriltags, key=score)


if __name__ == "__main__":
    tag_finder = AprilTagFinder()

    cap = cv2.VideoCapture(DisplayConfig.INPUT_PATH)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        apriltags = tag_finder.find_apriltags(frame)
        for apriltag in apriltags:
            for i in range(4):
                j = (i + 1) % 4
                point1 = (int(apriltag.getCorner(i).x),
                          int(apriltag.getCorner(i).y))
                point2 = (int(apriltag.getCorner(j).x),
                          int(apriltag.getCorner(j).y))
                cv2.line(frame, point1, point2, (0, 255, 0), 2)

            center_x = int(apriltag.getCenter().x)
            center_y = int(apriltag.getCenter().y)

            cv2.line(frame, (center_x - DisplayConfig.APRILTAG_CROSSHAIR_LINE_LENGTH, center_y),
                     (center_x + DisplayConfig.APRILTAG_CROSSHAIR_LINE_LENGTH, center_y), (0, 0, 255), 2)
            cv2.line(frame, (center_x, center_y - DisplayConfig.APRILTAG_CROSSHAIR_LINE_LENGTH),
                     (center_x, center_y + DisplayConfig.APRILTAG_CROSSHAIR_LINE_LENGTH), (0, 0, 255), 2)

            cv2.putText(frame, str(apriltag.getId()), (center_x + DisplayConfig.APRILTAG_CROSSHAIR_LINE_LENGTH,
                        center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.imshow(DisplayConfig.WINDOW_TITLE, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
