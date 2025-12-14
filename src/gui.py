"""
Copyright (c) FRC Team 3756 RamFerno.
Open Source Software; you can modify and/or share it under the terms of
the license viewable in the root directory of this project.
"""

import os
import cv2
import numpy as np
import pyautogui

from logs.logging_setup import setup_logger


################################################

class GUI:
    WINDOW_NAME = "JON GUI"

    def __init__(self):
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        self.logger = setup_logger(file_name)
        self.logger.info("Odometry logger initialized.")

        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.moveWindow(self.WINDOW_NAME, 0, 0)

        screen = pyautogui.size()
        self.screen_width = screen.width
        self.screen_height = int(screen.height * 0.9)

        cv2.resizeWindow(self.WINDOW_NAME, self.screen_width, int(self.screen_height * 0.4))

    def update(self, camera_frame, odometry_frame):
        """Update the display with both frames scaled to fill screen width."""
        if camera_frame is None or odometry_frame is None:
            self.logger.warning("[WARNING] One of the frames is empty, skipping frame update.")
            return

        screen = pyautogui.size()
        self.screen_width = screen.width
        self.screen_height = screen.height

        h_cam, w_cam = camera_frame.shape[:2]
        h_odo, w_odo = odometry_frame.shape[:2]

        if h_cam == 0 or h_odo == 0:
            print("[ERROR] Frame has zero height, skipping resize.")
            return

        pad = 10
        available_width = self.screen_width - pad

        aspect_cam = w_cam / h_cam
        aspect_odo = w_odo / h_odo

        target_height = available_width / (aspect_cam + aspect_odo)

        if target_height > self.screen_height:
            scale = self.screen_height / target_height
            target_height = self.screen_height
            available_width = int(available_width * scale)
            new_width_cam = int((available_width * aspect_cam) / (aspect_cam + aspect_odo))
            new_width_odo = int((available_width * aspect_odo) / (aspect_cam + aspect_odo))
        else:
            new_width_cam = int(target_height * aspect_cam)
            new_width_odo = int(target_height * aspect_odo)

        target_height = int(target_height)

        camera_resized = cv2.resize(camera_frame, (new_width_cam, target_height))
        odometry_resized = cv2.resize(odometry_frame, (new_width_odo, target_height))

        padding = np.full((target_height, pad, 3), 40, dtype=np.uint8)

        combined = np.hstack((camera_resized, padding, odometry_resized))

        if combined.shape[1] != self.screen_width:
            combined = cv2.resize(combined, (self.screen_width, target_height))

        cv2.resizeWindow(self.WINDOW_NAME, self.screen_width, target_height)

        cv2.imshow(self.WINDOW_NAME, combined)

    def close(self):
        """Close the OpenCV window cleanly."""
        cv2.destroyWindow(self.WINDOW_NAME)
