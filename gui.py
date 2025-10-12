import os
import cv2
import numpy as np

from config import CameraConfig
from logs.logging_setup import setup_logger

class GUI:
    WINDOW_NAME = "JON GUI"

    def __init__(self):
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        self.logger = setup_logger(file_name)
        self.logger.info("Odometry logger initialized.")

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(self.WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
        self.screen_width = int(cv2.getWindowImageRect(self.WINDOW_NAME)[2])
        self.screen_height = int(cv2.getWindowImageRect(self.WINDOW_NAME)[3])

    def update(self, camera_frame, odometry_frame):
        if camera_frame is None or odometry_frame is None:
            self.logger.warning("[WARNING] One of the frames is empty, skipping frame update.")
            return
        
        if self.screen_width == 0 or self.screen_height == 0:
            self.screen_width = 1920
            self.screen_height = 1080

        h_cam, w_cam = camera_frame.shape[:2]
        h_odo, w_odo = odometry_frame.shape[:2]

        if h_cam == 0 or h_odo == 0:
            print("[ERROR] Frame has zero height, skipping resize.")
            return

        aspect_ratio_cam = w_cam / h_cam
        aspect_ratio_odo = w_odo / h_odo

        new_width_cam = max(1, int(self.screen_height * aspect_ratio_cam))
        new_width_odo = max(1, int(self.screen_height * aspect_ratio_odo))

        camera_resized = cv2.resize(camera_frame, (new_width_cam, self.screen_height))
        odometry_resized = cv2.resize(odometry_frame, (new_width_odo, self.screen_height))

        pad = 10
        padding = np.full((CameraConfig.FRAME_HEIGHT_PX, pad, 3), 40, dtype=np.uint8)

        target_height = min(camera_resized.shape[0], odometry_resized.shape[0], padding.shape[0])
        camera_resized = cv2.resize(camera_resized, (camera_resized.shape[1], target_height))
        odometry_resized = cv2.resize(odometry_resized, (odometry_resized.shape[1], target_height))
        padding = np.full((target_height, pad, 3), 40, dtype=np.uint8)
        
        combined = np.hstack((camera_resized, padding, odometry_resized))

        if combined.shape[1] > self.screen_width:
            combined = cv2.resize(combined, (self.screen_width, self.screen_height))

        cv2.imshow(self.WINDOW_NAME, combined)

    def close(self):
        cv2.destroyWindow(self.WINDOW_NAME)
