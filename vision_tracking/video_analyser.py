import os
import logging
from typing import Tuple, List
from logs.logging_setup import setup_logger

import torch
import numpy as np
from ultralytics import YOLO

###############################################################

class YOLODetector:
    def __init__(self, weights_location: str, confidence_threshold: float) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        setup_logger(file_name)
        self.logger = logging.getLogger(file_name)

        self.device: torch.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.model: YOLO = YOLO(weights_location, task='detect')
        self.confidence_threshold: float = confidence_threshold

        self.logger.info(f'Model loaded to {self.device}')

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
