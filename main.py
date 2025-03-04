import os
import cv2
import math
import torch
import logging
import numpy as np
from time import time
from ultralytics import YOLO
from torchvision.ops import nms

from trackable_objects import Object
from camera_calculations.mono_video import MonoVision
from camera_calculations.stereo_video import StereoVision
from decision_engine.decision_matrix import DecisionMatrix
from networking.rio_communication import post_to_network_tables

###############################################################

script_name = os.path.splitext(os.path.basename(__file__))[0]

log_file = os.path.join("logs", f"{script_name}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Config:
    """Configuration settings for video processing."""
    ROTATE_IMAGE: bool = True
    FLIP_IMAGE_HORIZONTALLY: bool = False
    FLIP_IMAGE_VERTICALLY: bool = False

    FRAME_RATE_REFRESH_RATE: int = 200
    COVERAGE_THRESHOLD: float = 0.4
    CONFIDENCE_THRESHOLD: float = 0.7
    DISPLAY: bool = True
    SAVE_VIDEO: bool = True
    VIDEO_PATH: str = "video.mp4" #0 #"http://limelight.local:5800" #
    WEIGHTS_LOCATION: str = 'vision_tracking/runs/train/weights/best.onnx'
    LABEL_COLORS: dict[str, list[int]] = {
        "0": [0, 155, 255],
        "1": [0, 0, 255]
    }

class YOLODetector:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = YOLO(Config.WEIGHTS_LOCATION)

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

            if confidence >= Config.CONFIDENCE_THRESHOLD:
                boxes.append([x1, y1, x2, y2])
                confidences.append(confidence)
                class_ids.append(int(box.cls[0]))

        return np.array(boxes), np.array(confidences), np.array(class_ids)

class VideoDisplay:
    @staticmethod
    def show_frame(frame):
        cv2.imshow('Video', frame)

    @staticmethod
    def annotate_frame(frame, boxes, class_ids):
        """Annotate the frame with bounding boxes and labels."""
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            color = Config.LABEL_COLORS.get(str(class_ids[i]), (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        return frame

    @staticmethod
    def draw_angle_line(frame, angle):
        """Draws a line at a given angle from the bottom center of the screen."""
        height, width = frame.shape[:2]
        start_point = (width // 2, height - 1)
        length = 100
        end_x = int(start_point[0] + length * math.sin(math.radians(angle)))
        end_y = int(start_point[1] - length * math.cos(math.radians(angle)))
        
        if 0 <= end_x < width and 0 <= end_y < height:
            cv2.line(frame, start_point, (end_x, end_y), (0, 155, 255), 2)
        else:
            logging.warning(f"Line endpoint out of bounds: ({end_x}, {end_y}) for angle {angle}")

class FrameProcessor:
    def __init__(self):
        logging.info(f'Using device: {"GPU" if torch.cuda.is_available() else "CPU"}')
        self.detector = YOLODetector()
        self.decision_matrix = DecisionMatrix()
        self.property_calculation = MonoVision()
        self.depth_estimation = StereoVision()
        self.start_time = time()
        self.frame_count = 0
        self.notes = []

    def fix_frame(self, frame):
        if Config.ROTATE_IMAGE:
            frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if Config.FLIP_IMAGE_HORIZONTALLY:
            frame = cv2.flip(frame, 1)
        if Config.FLIP_IMAGE_VERTICALLY:
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
            
            frame = VideoDisplay.annotate_frame(frame, boxes_filtered, class_ids_filtered)
            self.update_notes(boxes_filtered, confidences_filtered, class_ids_filtered)

        if self.notes:
            note = self.decision_matrix.compute_best_game_piece(*self.notes)
        else:
            note = None
        return frame, note

    def update_notes(self, boxes, confidences, class_ids):
        """Update notes with detection data for game piece selection."""
        self.notes.clear()
        for box, conf, class_id in zip(boxes, confidences, class_ids):
            if class_id == 0:
                x1, y1, x2, y2 = box
                center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
                scale = ((x2 - x1) + (y2 - y1)) / 2
                ratio = (x2 - x1) / (y2 - y1)
                
                note = Object()
                note.update_frame_location(center_x, center_y, scale, ratio, time())
                note.update_confidence(conf)
                distance, angle = self.property_calculation.find_distance_and_angle(center_x, scale)
                note.update_relative_location(distance, angle)
                self.notes.append(note)

    def apply_nms(self, boxes, confidences):
        """Apply Non-Maximum Suppression to filter bounding boxes."""
        if boxes.size == 0:
            return np.array([])

        if self.detector.device.type == 'cuda':
            boxes_tensor = torch.tensor(boxes, dtype=torch.float16, device=self.detector.device)
            confidences_tensor = torch.tensor(confidences, dtype=torch.float16, device=self.detector.device)
        else:
            boxes_tensor = torch.tensor(boxes, dtype=torch.float32, device=self.detector.device)
            confidences_tensor = torch.tensor(confidences, dtype=torch.float32, device=self.detector.device)
        
        # Test removing
        indices = nms(boxes_tensor.to(self.detector.device), confidences_tensor.to(self.detector.device), Config.COVERAGE_THRESHOLD)
        return indices.cpu().numpy()

    def calculate_frame_rate(self):
        """Calculate and log the frame processing rate."""
        self.frame_count += 1
        if self.frame_count % Config.FRAME_RATE_REFRESH_RATE == 0:
            elapsed_time = time() - self.start_time
            fps = self.frame_count / elapsed_time
            logging.info(f"Processing FPS: {fps:.2f}")
            self.start_time = time()
            self.frame_count = 0

def main():
    processor = FrameProcessor()
    cap = cv2.VideoCapture(Config.VIDEO_PATH)
    
    if not cap.isOpened():
        logging.error(f"Error opening video: {Config.VIDEO_PATH}")
    
    if Config.SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
    
    try:
        print("Made connection to cap. Look for reaction on DS")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.warning("End of video stream.")
                break

            frame = processor.fix_frame(frame)
            
            processed_frame, note = processor.process_frame(frame)
            if note:
                post_to_network_tables((note.distance, note.angle))
                
            if Config.DISPLAY:
                if note:
                    VideoDisplay.draw_angle_line(frame, note.angle)
                VideoDisplay.show_frame(processed_frame)
                
            if Config.SAVE_VIDEO:
                out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                
            processor.calculate_frame_rate()

            if Config.DISPLAY & cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if Config.SAVE_VIDEO:
            out.release()
            logging.info("Video file closed properly.")
        cv2.destroyAllWindows()

###############################################################

if __name__ == "__main__":
    main()