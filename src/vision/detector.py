import os
import logging

import platform
import re
from typing import Tuple, List

import torch
import numpy as np
from ultralytics.models import YOLO

from config import DetectorConfig
from logs.logging_setup import setup_logger

###############################################################

class Detector:
    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        setup_logger(file_name)
        self.logger = logging.getLogger(file_name)
        
        self.device: torch.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu')
        
        model_path = None
        if self.device.type == 'cuda':
            custom_tensorrt_path = DetectorConfig.WEIGHTS_LOCATION + f'best_{self.get_system_identifier()}.engine'
            if os.path.exists(custom_tensorrt_path):
                model_path = custom_tensorrt_path
            else:
                logging.warning(f'Could not find custom trained {custom_tensorrt_path}')
        
        if not model_path:
            model_path = DetectorConfig.WEIGHTS_LOCATION + f'best.onnx'

        self.model: YOLO = YOLO(model_path, task='detect')

        self.logger.info(f'{model_path} Model loaded to {self.device}')
        
    def get_system_identifier(self) -> str:
        system = platform.system().lower()
        arch = platform.machine().lower()
        if arch in ["amd64"]:
            arch = "x86_64"

        gpu_name = torch.cuda.get_device_name(0)
        gpu_name = re.sub(r'[^a-zA-Z0-9]', '', gpu_name).lower()

        return f"{system}_{arch}_{gpu_name}"

    def detect(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run detection on a frame and return processed results."""
        with torch.no_grad():
            results = self.model.predict(frame, half=True)[0]
        return self.extract_detections(results)

    def extract_detections(self, results) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extract bounding boxes, confidences, and class IDs."""
        boxes: List[List[int]] = []
        confidences: List[float] = []
        class_ids: List[int] = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            confidence: float = float(box.conf[0])

            if confidence >= DetectorConfig.CONFIDENCE_THRESHOLD:
                boxes.append([x1, y1, x2, y2])
                confidences.append(confidence)
                class_ids.append(int(box.cls[0]))

        return np.array(boxes), np.array(confidences), np.array(class_ids)
