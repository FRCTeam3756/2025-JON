"""
Copyright (c) FRC Team 3756 RamFerno.
Open Source Software; you can modify and/or share it under the terms of
the license viewable in the root directory of this project.
"""

import os
import re
import logging
import platform
import time
from typing import Optional, Tuple, List
from queue import Queue, Empty
from threading import Lock, Thread

import torch
import numpy as np
from ultralytics.models import YOLO

from constants.vision import DetectorConfig
from logs.logging_setup import setup_logger

###############################################################

class YOLODetector:
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

################################################

class AsyncYOLODetector(YOLODetector):
    def __init__(self):
        super().__init__()
        self.frame_queue = Queue(maxsize=1)
        self.result_lock = Lock()
        self.latest_detections: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
        self.running = True
        self.thread = Thread(target=self._process_loop, daemon=True)
        self.thread.start()

    def _process_loop(self):
        while self.running:
            try:
                frame = self.frame_queue.get(timeout=0.05)
            except Empty:
                continue

            try:
                boxes, confidences, class_ids = super().detect(frame)
                result = (boxes, confidences, class_ids)
                with self.result_lock:
                    self.latest_detections = result
            except Exception as e:
                print(f"[AsyncDetector] Error during detection: {e}")
                time.sleep(0.01)

    def submit_frame(self, frame: np.ndarray):
        if not self.running:
            return
        
        if self.frame_queue.full():
            try:
                self.frame_queue.get_nowait()
            except:
                pass
        self.frame_queue.put(frame)

    def get_latest_detections(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with self.result_lock:
            if self.latest_detections is not None:
                return self.latest_detections
            else:
                return np.empty((0, 4), dtype=int), np.empty((0,), dtype=float), np.empty((0,), dtype=int)

    def stop(self):
        self.running = False
        self.thread.join()
