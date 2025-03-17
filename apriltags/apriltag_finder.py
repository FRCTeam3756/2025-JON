import os
import logging
from logs.logging_setup import setup_logger
from typing import List

import cv2
import robotpy_apriltag as apriltag

from config import CameraConfig, AprilTagConfig

class AprilTagFinder:
    def __init__(self):
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        setup_logger(file_name)
        self.logger = logging.getLogger(file_name)

        self.detector = apriltag.AprilTagDetector()
        self.detector.addFamily("tag36h11", 3)\
    
    @staticmethod
    def estimate_distance(tag_size_pixels):
        """Estimate distance to the tag based on its size in the image."""
        return (CameraConfig.FOCAL_LENGTH * AprilTagConfig.APRILTAG_SIZE_IN_CM) / tag_size_pixels

    def find_apriltags(self, frame) -> List:
        """Main loop for processing frames and sending drive instructions."""
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detections = self.detector.detect(gray_frame)
        
        return detections