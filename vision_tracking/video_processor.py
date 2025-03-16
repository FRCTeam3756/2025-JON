import os
import time
import logging
from typing import Tuple, Dict, List, Type, Union

import cv2
import torch
import torchvision
import numpy as np
from ultralytics import YOLO

from config import LoggingConfig, DisplayConfig, YOLOConfig
from .video_display import VideoDisplay
from decision_engine.trackable_objects import Algae, Cage, CagePole, Chain, Coral, Robot
from camera_calculations.mono_video import MonoVision
from camera_calculations.stereo_video import StereoVision

###############################################################

script_name = os.path.splitext(os.path.basename(__file__))[0]
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_file = os.path.join(log_dir, f"{script_name}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class YOLODetector:
    def __init__(self, weights_location: str, confidence_threshold: float) -> None:
        self.device: torch.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: YOLO = YOLO(weights_location, task='detect')
        self.confidence_threshold: float = confidence_threshold

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run detection on a frame and return processed results."""
        with torch.no_grad():
            results = self.model.predict(frame)[0]
        return self.extract_detections(results)

    def extract_detections(self, results) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract bounding boxes, confidences, and class IDs."""
        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for box in results.boxes:
            # print(box.xywh.cpu()) // Add to JON
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence: float = float(box.conf[0])

            if confidence >= self.confidence_threshold:
                boxes.append([x1, y1, x2, y2])
                confidences.append(confidence)
                class_ids.append(int(box.cls[0]))

        return np.array(boxes), np.array(confidences), np.array(class_ids)

class FrameProcessor:
    def __init__(self) -> None:
        logging.info(f'Using device: {"GPU" if torch.cuda.is_available() else "CPU"}')
        self.detector: YOLODetector = YOLODetector(YOLOConfig.WEIGHTS_LOCATION, YOLOConfig.CONFIDENCE_THRESHOLD)
        self.property_calculation: MonoVision = MonoVision()
        self.depth_estimation: StereoVision = StereoVision()
        self.start_time: float = time.time()
        self.frame_count: int = 0
        self.game_pieces: Dict[Type[Union[Algae, Cage, CagePole, Chain, Coral, Robot]], List] = {
            obj: [] for obj in (Algae, Cage, CagePole, Chain, Coral, Robot)
        }

    def transform_frame(self, frame: np.ndarray) -> np.ndarray:
        if DisplayConfig.ROTATE_IMAGE:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if DisplayConfig.FLIP_IMAGE_HORIZONTALLY:
            frame = cv2.flip(frame, 1)
        if DisplayConfig.FLIP_IMAGE_VERTICALLY:
            frame = cv2.flip(frame, 0)

        return frame

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Dict[Type[Union[Algae, Cage, CagePole, Chain, Coral, Robot]], List]]:
        """Processes a single frame for detections and annotations."""
        boxes, confidences, class_ids = self.detector.detect(frame)
        
        if boxes.size > 0:
            indices = self.apply_nms(boxes, confidences)
            
            boxes_filtered, confidences_filtered, class_ids_filtered = boxes[indices], confidences[indices], class_ids[indices]
            
            frame = VideoDisplay.annotate_frame(frame, boxes_filtered, class_ids_filtered, DisplayConfig.LABEL_COLOURS)
            self.update_game_pieces(boxes_filtered, confidences_filtered, class_ids_filtered)

        return frame, self.game_pieces

    def extract_features(self, box: List[int]) -> Tuple[int, int, float, float]:
        """Extract common object features."""
        x1, y1, x2, y2 = box
        center_x: int = (x1 + x2) // 2
        center_y: int = (y1 + y2) // 2
        scale: float = ((x2 - x1) + (y2 - y1)) / 2
        ratio: float = (x2 - x1) / (y2 - y1)
        return center_x, center_y, scale, ratio

    def update_game_pieces(self, boxes: np.ndarray, confidences: np.ndarray, class_ids: np.ndarray) -> None:
        """Update game pieces with detection data for game piece selection."""
        for key in self.game_pieces:
            self.game_pieces[key] = []

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            match class_id:
                case 0: # Algae
                    center_x, center_y, scale, ratio = self.extract_features(box)
                    
                    algae = Algae()
                    algae.update_frame_location(center_x, center_y, scale, ratio, time.time())
                    algae.update_confidence(conf)
                    distance, angle = self.property_calculation.find_distance_and_angle(center_x, scale)
                    algae.update_relative_location(distance, angle)
                    self.game_pieces[Algae].append(algae)
                
                case 1: # Cage
                    center_x, center_y, scale, ratio = self.extract_features(box)
                    
                    cage = Cage()
                    cage.update_frame_location(center_x, center_y, scale, ratio, time.time())
                    cage.update_confidence(conf)
                    distance, angle = self.property_calculation.find_distance_and_angle(center_x, scale)
                    cage.update_relative_location(distance, angle)
                    self.game_pieces[Cage].append(cage)

                case 2: # Cage Pole
                    center_x, center_y, scale, ratio = self.extract_features(box)
                    
                    cage_pole = CagePole()
                    cage_pole.update_frame_location(center_x, center_y, scale, ratio, time.time())
                    cage_pole.update_confidence(conf)
                    distance, angle = self.property_calculation.find_distance_and_angle(center_x, scale)
                    cage_pole.update_relative_location(distance, angle)
                    self.game_pieces[CagePole].append(cage_pole)
                
                case 3: # Chain
                    center_x, center_y, scale, ratio = self.extract_features(box)
                    
                    chain = Chain()
                    chain.update_frame_location(center_x, center_y, scale, ratio, time.time())
                    chain.update_confidence(conf)
                    distance, angle = self.property_calculation.find_distance_and_angle(center_x, scale)
                    chain.update_relative_location(distance, angle)
                    self.game_pieces[Chain].append(chain)
                
                case 4: # Coral
                    center_x, center_y, scale, ratio = self.extract_features(box)
                    
                    coral = Coral()
                    coral.update_frame_location(center_x, center_y, scale, ratio, time.time())
                    coral.update_confidence(conf)
                    distance, angle = self.property_calculation.find_distance_and_angle(center_x, scale)
                    coral.update_relative_location(distance, angle)
                    self.game_pieces[Coral].append(coral)
                
                case 5: # Robot
                    center_x, center_y, scale, ratio = self.extract_features(box)
                    
                    robot = Robot()
                    robot.update_frame_location(center_x, center_y, scale, ratio, time.time())
                    robot.update_confidence(conf)
                    distance, angle = self.property_calculation.find_distance_and_angle(center_x, scale)
                    robot.update_relative_location(distance, angle)
                    self.game_pieces[Robot].append(robot)
                case _: continue

    def apply_nms(self, boxes: np.ndarray, confidences: np.ndarray) -> np.ndarray:
        """Apply Non-Maximum Suppression to filter bounding boxes."""
        if len(boxes) == 0:
            return np.array([])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=self.detector.device)
        confidences_tensor = torch.tensor(confidences, dtype=torch.float32, device=self.detector.device)
        
        indices = torchvision.ops.nms(boxes_tensor.to(self.detector.device), confidences_tensor.to(self.detector.device), YOLOConfig.IOU_THRESHOLD)
        return indices.cpu().numpy()

    def calculate_frame_rate(self) -> None:
        """Calculate and log the frame processing rate."""
        self.frame_count += 1
        if self.frame_count % LoggingConfig.FPS_LOGGING_RATE == 0:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time
            logging.info(f"Processing FPS: {fps:.2f}")
            self.start_time = time.time()
            self.frame_count = 0