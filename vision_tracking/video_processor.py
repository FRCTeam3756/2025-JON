import os
import cv2
import logging
from time import time

import torch
import torchvision
import numpy as np
from ultralytics import YOLO

from .video_display import VideoDisplay
from decision_engine.trackable_objects import *
from camera_calculations.mono_video import MonoVision
from camera_calculations.stereo_video import StereoVision

###############################################################

script_name = os.path.splitext(os.path.basename(__file__))[0]

log_file = os.path.join("logs", f"{script_name}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class YOLODetector:
    def __init__(self, weights_location, confidence_threshold):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(weights_location, task='detect')
        self.confidence_threshold = confidence_threshold

    def detect(self, frame):
        """Run detection on a frame and return processed results."""
        with torch.no_grad():
            results = self.model.predict(frame)[0]
        return self.extract_detections(results)

    def extract_detections(self, results):
        """Extract bounding boxes, confidences, and class IDs."""
        boxes, confidences, class_ids = [], [], []
        for box in results.boxes:
            # print(box.xywh.cpu()) // Add to JON
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence = float(box.conf[0])

            if confidence >= self.confidence_threshold:
                boxes.append([x1, y1, x2, y2])
                confidences.append(confidence)
                class_ids.append(int(box.cls[0]))

        return np.array(boxes), np.array(confidences), np.array(class_ids)

class FrameProcessor:
    def __init__(self, config):
        logging.info(f'Using device: {"GPU" if torch.cuda.is_available() else "CPU"}')
        self.config = config
        self.detector = YOLODetector(self.config.WEIGHTS_LOCATION, self.config.CONFIDENCE_THRESHOLD)
        self.property_calculation = MonoVision()
        self.depth_estimation = StereoVision()
        self.start_time = time()
        self.frame_count = 0
        self.game_pieces = {
            Algae: [],
            Cage: [],
            CagePole: [],
            Chain: [],
            Coral: [],
            Robot: []
        }

    def fix_frame(self, frame):
        if self.config.ROTATE_IMAGE:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.config.FLIP_IMAGE_HORIZONTALLY:
            frame = cv2.flip(frame, 1)
        if self.config.FLIP_IMAGE_VERTICALLY:
            frame = cv2.flip(frame, 0)

        return frame

    def process_frame(self, frame):
        """Processes a single frame for detections and annotations."""
        boxes, confidences, class_ids = self.detector.detect(frame)
        
        if boxes.size > 0:
            indices = self.apply_nms(boxes, confidences)
            
            boxes_filtered = boxes[indices]
            confidences_filtered = confidences[indices]
            class_ids_filtered = class_ids[indices]
            
            frame = VideoDisplay.annotate_frame(frame, boxes_filtered, class_ids_filtered, self.config.LABEL_COLOURS)
            self.update_game_pieces(boxes_filtered, confidences_filtered, class_ids_filtered)

        return frame, self.game_pieces

    def update_game_pieces(self, boxes, confidences, class_ids):
        """Update game pieces with detection data for game piece selection."""
        for key in self.game_pieces:
            self.game_pieces[key] = []

        for box, conf, class_id in zip(boxes, confidences, class_ids):
            match class_id:
                case 0: # Algae
                    x1, y1, x2, y2 = box
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    scale = ((x2 - x1) + (y2 - y1)) / 2
                    ratio = (x2 - x1) / (y2 - y1)
                    
                    alga = Algae()
                    alga.update_frame_location(center_x, center_y, scale, ratio, time())
                    alga.update_confidence(conf)
                    distance, angle = self.property_calculation.find_distance_and_angle(center_x, scale)
                    alga.update_relative_location(distance, angle)
                    self.game_pieces[Algae].append(alga)
                
                case 1: # Cage
                    x1, y1, x2, y2 = box
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    scale = ((x2 - x1) + (y2 - y1)) / 2
                    ratio = (x2 - x1) / (y2 - y1)
                    
                    cage = Cage()
                    cage.update_frame_location(center_x, center_y, scale, ratio, time())
                    cage.update_confidence(conf)
                    distance, angle = self.property_calculation.find_distance_and_angle(center_x, scale)
                    cage.update_relative_location(distance, angle)
                    self.game_pieces[Cage].append(cage)

                case 2: # Cage Pole
                    x1, y1, x2, y2 = box
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    scale = ((x2 - x1) + (y2 - y1)) / 2
                    ratio = (x2 - x1) / (y2 - y1)
                    
                    cage_pole = CagePole()
                    cage_pole.update_frame_location(center_x, center_y, scale, ratio, time())
                    cage_pole.update_confidence(conf)
                    distance, angle = self.property_calculation.find_distance_and_angle(center_x, scale)
                    cage_pole.update_relative_location(distance, angle)
                    self.game_pieces[CagePole].append(cage_pole)
                
                case 3: # Chain
                    x1, y1, x2, y2 = box
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    scale = ((x2 - x1) + (y2 - y1)) / 2
                    ratio = (x2 - x1) / (y2 - y1)
                    
                    chain = Chain()
                    chain.update_frame_location(center_x, center_y, scale, ratio, time())
                    chain.update_confidence(conf)
                    distance, angle = self.property_calculation.find_distance_and_angle(center_x, scale)
                    chain.update_relative_location(distance, angle)
                    self.game_pieces[Chain].append(chain)
                
                case 4: # Coral
                    x1, y1, x2, y2 = box
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    scale = ((x2 - x1) + (y2 - y1)) / 2
                    ratio = (x2 - x1) / (y2 - y1)
                    
                    coral = Coral()
                    coral.update_frame_location(center_x, center_y, scale, ratio, time())
                    coral.update_confidence(conf)
                    distance, angle = self.property_calculation.find_distance_and_angle(center_x, scale)
                    coral.update_relative_location(distance, angle)
                    self.game_pieces[Coral].append(coral)
                
                case 5: # Robot
                    x1, y1, x2, y2 = box
                    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                    scale = ((x2 - x1) + (y2 - y1)) / 2
                    ratio = (x2 - x1) / (y2 - y1)
                    
                    robot = Robot()
                    robot.update_frame_location(center_x, center_y, scale, ratio, time())
                    robot.update_confidence(conf)
                    distance, angle = self.property_calculation.find_distance_and_angle(center_x, scale)
                    robot.update_relative_location(distance, angle)
                    self.game_pieces[Robot].append(robot)

    def apply_nms(self, boxes, confidences):
        """Apply Non-Maximum Suppression to filter bounding boxes."""
        if len(boxes) == 0:
            return np.array([])

        boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=self.detector.device)
        confidences_tensor = torch.tensor(confidences, dtype=torch.float32, device=self.detector.device)
        
        indices = torchvision.ops.nms(boxes_tensor.to(self.detector.device), confidences_tensor.to(self.detector.device), self.config.COVERAGE_THRESHOLD)
        
        return indices.cpu().numpy()

    def calculate_frame_rate(self):
        """Calculate and log the frame processing rate."""
        self.frame_count += 1
        if self.frame_count % self.config.MAXIMUM_FRAME_RATE == 0:
            elapsed_time = time() - self.start_time
            fps = self.frame_count / elapsed_time
            logging.info(f"Processing FPS: {fps:.2f}")
            self.start_time = time()
            self.frame_count = 0