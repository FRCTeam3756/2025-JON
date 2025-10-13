import os
import time
import logging
from logs.logging_setup import setup_logger
from typing import Tuple, List

import cv2
import torch
import torchvision
import numpy as np

from config import AutoAlgaeConfig, AutoCoralConfig, AutoHangConfig, AutoRobotConfig, DetectorConfig, DisplayConfig, LoggingConfig
from src.vision.detector import Detector
from src.display import Display
from src.apriltags.apriltags import AprilTagDetection, AprilTagFinder
from src.navigator.trackable_objects import Algae, Cage, Coral, GamePieces, Robot
from src.camera.monovision import MonoVision

###############################################################


class Processor:
    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        setup_logger(file_name)
        self.logger = logging.getLogger(file_name)

        self.logger.info(
            f'Using device: {"GPU" if torch.cuda.is_available() else "MPS" if torch.mps.is_available() else "CPU"}')
        self.yolo_detector = Detector()
        self.apriltag_detector = AprilTagFinder()
        self.start_time = time.time()
        self.frame_count = 0
        self.game_pieces = GamePieces()

    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        if DisplayConfig.ROTATE_IMAGE:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if DisplayConfig.FLIP_IMAGE_HORIZONTALLY:
            frame = cv2.flip(frame, 1)
        if DisplayConfig.FLIP_IMAGE_VERTICALLY:
            frame = cv2.flip(frame, 0)

        return frame

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, GamePieces, List[AprilTagDetection]]:
        """Processes a single frame for detections and annotations."""
          # Find apriltags before resizing for better accuracy
        apriltags = self.apriltag_detector.detect_tags(
            frame)  
        frame = cv2.resize(
            frame, (DisplayConfig.FRAME_WIDTH_PX, DisplayConfig.FRAME_HEIGHT_PX))
        boxes, confidences, class_ids = self.yolo_detector.detect(frame)

        if boxes.size > 0:
            indices = self.apply_nms(boxes, confidences)

            boxes_filtered, confidences_filtered, class_ids_filtered = boxes[
                indices], confidences[indices], class_ids[indices]

            frame = Display.annotate_frame(
                frame, boxes_filtered, class_ids_filtered, apriltags)
            self.update_game_pieces(
                boxes_filtered, confidences_filtered, class_ids_filtered)

        return frame, self.game_pieces, apriltags

    def extract_features(self, x1: int, y1: int, x2: int, y2: int) -> Tuple[int, int, float, float]:
        """Extract object features."""
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        scale = ((x2 - x1) + (y2 - y1)) / 2
        ratio = (x2 - x1) / (y2 - y1)
        return center_x, center_y, scale, ratio

    def update_game_pieces(self, boxes: np.ndarray, confidences: np.ndarray, class_ids: np.ndarray) -> None:
        """Update game pieces with detection data for game piece selection."""
        self.game_pieces.clear()

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            x1, y1, x2, y2 = box
            center_x, center_y, scale, ratio = self.extract_features(
                x1, y1, x2, y2)
            object_width_px = x2 - x1

            match class_id:
                case 0:  # Algae
                    algae = Algae()
                    algae.update_frame_location(
                        center_x, center_y, scale, ratio, time.time())
                    algae.update_confidence(conf)
                    distance = MonoVision.get_distance_to_object_in_mm(
                        AutoAlgaeConfig.ALGAE_SIZE_MM, object_width_px, DisplayConfig.FRAME_WIDTH_PX)
                    angle = MonoVision.get_angle_to_object_in_degrees(center_x, DisplayConfig.FRAME_WIDTH_PX)
                    algae.update_relative_location(distance, angle)
                    self.game_pieces.add(Algae, algae)

                case 1:  # Cage
                    cage = Cage()
                    cage.update_frame_location(
                        center_x, center_y, scale, ratio, time.time())
                    cage.update_confidence(conf)
                    distance = MonoVision.get_distance_to_object_in_mm(
                        AutoHangConfig.CAGE_WIDTH_MM, object_width_px, DisplayConfig.FRAME_WIDTH_PX)
                    angle = MonoVision.get_angle_to_object_in_degrees(center_x, DisplayConfig.FRAME_WIDTH_PX)
                    cage.update_relative_location(distance, angle)
                    self.game_pieces.add(Cage, cage)

                case 2:  # Coral
                    coral = Coral()
                    coral.update_frame_location(
                        center_x, center_y, scale, ratio, time.time())
                    coral.update_confidence(conf)
                    distance = MonoVision.get_distance_to_object_in_mm(
                        AutoCoralConfig.CORAL_SIZE_MM, object_width_px, DisplayConfig.FRAME_WIDTH_PX)
                    angle = MonoVision.get_angle_to_object_in_degrees(center_x, DisplayConfig.FRAME_WIDTH_PX)
                    coral.update_relative_location(distance, angle)
                    self.game_pieces.add(Coral, coral)

                case 3:  # Robot
                    robot = Robot()
                    robot.update_frame_location(
                        center_x, center_y, scale, ratio, time.time())
                    robot.update_confidence(conf)
                    distance = MonoVision.get_distance_to_object_in_mm(
                        AutoRobotConfig.AVERAGE_ROBOT_SIZE_MM, object_width_px, DisplayConfig.FRAME_WIDTH_PX)
                    angle = MonoVision.get_angle_to_object_in_degrees(center_x, DisplayConfig.FRAME_WIDTH_PX)
                    robot.update_relative_location(distance, angle)
                    self.game_pieces.add(Robot, robot)

                case _: continue

    def apply_nms(self, boxes: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """Apply Non-Maximum Suppression to filter bounding boxes."""
        if len(boxes) == 0:
            return np.array([])

        boxes_tensor = torch.tensor(
            boxes, dtype=torch.float32, device=self.yolo_detector.device)
        confidences_tensor = torch.tensor(
            confidences, dtype=torch.float32, device=self.yolo_detector.device)

        indices = torchvision.ops.nms(boxes_tensor.to(self.yolo_detector.device), confidences_tensor.to(
            self.yolo_detector.device), DetectorConfig.IOU_THRESHOLD)
        return indices.cpu().numpy()

    def calculate_frame_rate(self) -> None:
        """Calculate and log the frame processing rate."""
        self.frame_count += 1
        if self.frame_count % LoggingConfig.FPS_LOGGING_RATE == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            self.logger.info(f"Processing FPS: {fps:.2f}")
            self.start_time = time.time()
            self.frame_count = 0
