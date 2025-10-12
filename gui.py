import cv2
import numpy as np

from config import CameraConfig

class GUI:
    def __init__(self):
        cv2.namedWindow("JON GUI", cv2.WINDOW_NORMAL)

    def update(self, camera_frame, odometry_frame):
        aspect_ratio = odometry_frame.shape[1] / odometry_frame.shape[0]
        new_width = int(CameraConfig.FRAME_HEIGHT_PX * aspect_ratio)
        odo = cv2.resize(odometry_frame, (new_width, CameraConfig.FRAME_HEIGHT_PX))

        pad = 10
        padding = np.full((CameraConfig.FRAME_HEIGHT_PX, pad, 3), 40, dtype=np.uint8)  # gray divider
        combined = np.hstack((camera_frame, padding, odo))

        cv2.imshow("JON GUI", combined)

    def close(self):
        cv2.destroyWindow("JON GUI")
