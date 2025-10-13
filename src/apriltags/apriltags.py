import cv2
import math
import numpy as np
from pupil_apriltags import Detector
from typing import List, Optional

from config import AprilTagConfig, CameraConfig, DisplayConfig


class AprilTagDetection:
    def __init__(self, detection, scale: float):
        self.detection = detection
        self.tag_id = detection.tag_id
        self.pose_t = detection.pose_t

        if detection.center is not None:
            self.center = [c / scale for c in detection.center]
        else:
            self.center = None

        if detection.corners is not None:
            self.corners = [[x / scale, y / scale] for x, y in detection.corners]
        else:
            self.corners = None

    @property
    def center_x(self) -> Optional[float]:
        return self.center[0] if self.center is not None else None

    @property
    def center_y(self) -> Optional[float]:
        return self.center[1] if self.center is not None else None
    
    @property
    def id(self):
        return self.tag_id
    
    @property
    def relative_distance(self) -> Optional[float]:
        pose = self.pose_t.flatten()
        distance = np.linalg.norm(pose)
        return float(distance)
    
    @property
    def relative_angle(self) -> Optional[float]:
        if self.center_x:
            return math.degrees(math.atan((self.center_x - (CameraConfig.NATIVE_FRAME_WIDTH_PX / 2)) / (CameraConfig.NATIVE_FRAME_WIDTH_PX / (2 * math.tan(math.radians(CameraConfig.HORIZONTAL_FOV_DEG) / 2)))))
        else:
            return None

    def corner_x(self, index: int) -> Optional[float]:
        """Return the x-coordinate of corner 0-3."""
        if self.corners is None or not (0 <= index <= 3):
            return None
        return self.corners[index][0]

    def corner_y(self, index: int) -> Optional[float]:
        """Return the y-coordinate of corner 0-3."""
        if self.corners is None or not (0 <= index <= 3):
            return None
        return self.corners[index][1]
    
class AprilTagFinder:
    def __init__(self) -> None:
        self.detector = Detector(
            families=AprilTagConfig.TAG_FAMILY,
            nthreads=1,
            quad_decimate=1.0,  # no downscaling
            quad_sigma=1.0,     # blur for noise tolerance
            refine_edges=True,
            decode_sharpening=0.1,
            debug=False
        )

    @staticmethod
    def preprocess(frame: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        return clahe.apply(gray)

    def detect_tags(self, frame: np.ndarray) -> List[AprilTagDetection]:
        gray = self.preprocess(frame)
        detections = []
        for scale in [1.0, 0.75, 0.5]:
            scaled = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            scaled = scaled.astype(np.uint8)
            tags = self.detector.detect(
                scaled,
                estimate_tag_pose=True,
                camera_params=(CameraConfig.FOCAL_LENGTH_PX, CameraConfig.FOCAL_LENGTH_PX,
                               CameraConfig.NATIVE_FRAME_WIDTH_PX / 2, CameraConfig.NATIVE_FRAME_HEIGHT_PX / 2),
                tag_size=(AprilTagConfig.APRILTAG_SIZE_MM / 1000)
            )
            tags = tags if isinstance(tags, list) else [tags]
            for tag in tags:
                if tag.center is None or tag.corners is None:
                    continue
                detections.append(AprilTagDetection(tag, scale))
        detections = self.deduplicate(detections)
        detections = self.normalize_tags(detections)
        return detections
    
    def normalize_tags(self, tags: List[AprilTagDetection]):
        scale_x = DisplayConfig.FRAME_WIDTH_PX / CameraConfig.NATIVE_FRAME_WIDTH_PX
        scale_y = DisplayConfig.FRAME_HEIGHT_PX / CameraConfig.NATIVE_FRAME_HEIGHT_PX
        for tag in tags:
            if tag.center:
                tag.center = [tag.center[0] * scale_x, tag.center[1] * scale_y]
            if tag.corners:
                tag.corners = [[x * scale_x, y * scale_y] for x, y in tag.corners]
        return tags
    
    @staticmethod
    def deduplicate(tags: List[AprilTagDetection]):
        """Remove duplicate detections of the same tag ID (from multiscale scans)."""
        unique = {}
        for tag in tags:
            if not tag.corners:
                continue
            
            size = np.mean([
                np.linalg.norm(np.array(tag.corners[i]) - np.array(tag.corners[(i + 1) % 4]))
                for i in range(4)
            ])
            if tag.tag_id not in unique or size > unique[tag.tag_id][1]:
                unique[tag.tag_id] = (tag, size)
        return [v[0] for v in unique.values()]

    @staticmethod
    def get_best_tag(tags: List) -> Optional[AprilTagDetection]:
        if not tags:
            return None
        cx, cy = CameraConfig.NATIVE_FRAME_WIDTH_PX / 2, CameraConfig.NATIVE_FRAME_HEIGHT_PX / 2
        def score(t):
            size = np.mean([np.linalg.norm(np.array(t.corners[i]) - np.array(t.corners[(i + 1) % 4]))
                            for i in range(4)])
            offset = np.linalg.norm(np.array(t.center) - np.array([cx, cy]))
            return size - 0.5 * offset
        return max(tags, key=score)