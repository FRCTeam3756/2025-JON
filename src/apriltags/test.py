"""
Copyright (c) FRC Team 3756 RamFerno.
Open Source Software; you can modify and/or share it under the terms of
the license viewable in the root directory of this project.
"""

import cv2

from constants.monitoring.display import DisplayConfig
from src.display import Display
from src.apriltags.apriltags import AprilTagFinder

###############################################################

if __name__ == "__main__":
    cap = cv2.VideoCapture(DisplayConfig.INPUT_PATH)
    finder = AprilTagFinder()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        tags = finder.detect_tags(frame)
        best = finder.get_best_tag(tags)
        for tag in tags:
            Display.draw_apriltag(frame, tag)

        cv2.imshow("AprilTag Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
