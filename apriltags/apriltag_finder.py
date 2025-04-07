import os
import math
import logging
from logs.logging_setup import setup_logger
from typing import List

import cv2
import robotpy_apriltag as apriltag
from robotpy_apriltag import AprilTagDetection

from config import CameraConfig, AprilTagConfig

class AprilTagFinder:
    def __init__(self):
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        setup_logger(file_name)
        self.logger = logging.getLogger(file_name)

        self.apriltag_detector = apriltag.AprilTagDetector()
        self.apriltag_detector.addFamily("tag36h11", 3)
    
    @staticmethod
    def estimate_distance(apriltag: AprilTagDetection):
        """Estimate distance to the tag based on its size in the image."""
        apriltag_width_in_pixels = apriltag.getCorners[0] - apriltag.getCorners[4]
        return (CameraConfig.FOCAL_LENGTH * AprilTagConfig.APRILTAG_SIZE_IN_CM) / apriltag_width_in_pixels
    
    @staticmethod
    def calculate_anglular_diviation(apriltag: AprilTagDetection):
        """Calculate the angle offset of an apriltag."""
        return math.degrees(math.atan((apriltag.getCenter().x - (CameraConfig.FRAME_WIDTH / 2)) / CameraConfig.FOCAL_LENGTH))

    def find_apriltags(self, frame) -> List:
        """Main loop for processing frames and sending drive instructions."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.apriltag_detector.detect(gray_frame)