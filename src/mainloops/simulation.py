

from logging import Logger
from typing import List, Optional

import cv2

from apriltags.apriltags import AprilTagDetection, AprilTagFinder
from camera.monovision import MonoVision
from config import DebugConfig, DisplayConfig
from src.display import Display
from src.gui import GUI
from localization.localization import Localization
from navigator.autoalgae import AlgaePickupCommand
from navigator.autocoral import CoralPickupCommand
from navigator.autoprocessor import ProcessorScoringCommand
from navigator.autoreef import ReefScoringCommand
from navigator.trackable_objects import Algae
from odometry.odometry import Odometry
from vision.processor import Processor


def simulation(logger: Logger, gui: GUI, odometry: Odometry, localization: Localization, frame_processor: Processor, autoalgae: AlgaePickupCommand, autocoral: CoralPickupCommand, autoreef: ReefScoringCommand, autoprocessor: ProcessorScoringCommand, cap: cv2.VideoCapture, out: cv2.VideoWriter) -> None:
    logger.info("Running in TESTING mode.")

    robot_x_m, robot_y_m, robot_heading_rad = 8, 2, 2.9
    current_key: Optional[str] = None
    messages: List = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream.")
            break

        frame = frame_processor.transform_frame(frame)
        camera_frame, visible_game_pieces, apriltags = frame_processor.process_frame(
            frame)
        frame_processor.calculate_frame_rate()

        keycode = cv2.waitKey(1) & 0xFF
        if keycode != 255:  # no key pressed
            key_char = chr(keycode)
            if key_char in DebugConfig.TASK_KEYS:
                current_key = key_char
        if not current_key:
            current_key = DebugConfig.DEFAULT_KEY

        processor_id = 3
        reef_ids = {6, 7, 8, 9, 10, 11}

        processor_apriltag: Optional[AprilTagDetection] = next(
            (tag for tag in apriltags if tag.id == processor_id),
            None
        )
        reef_apriltags: Optional[List[AprilTagDetection]] = [
            tag for tag in apriltags if tag.id in reef_ids
        ]
        closest_apriltag = AprilTagFinder.get_best_tag(apriltags)
        if closest_apriltag and closest_apriltag.relative_distance and closest_apriltag.relative_angle:
            robot_x_m, robot_y_m, robot_heading_rad = localization.get_world_position(closest_apriltag.id, closest_apriltag.relative_distance, closest_apriltag.relative_angle)
        odometry.game_pieces.add(visible_game_pieces)
        logger.debug(
            f'[DEBUG]: Sending {len(visible_game_pieces.get_all())} objects to odometry')
        odometry_frame = odometry.process_frame(
            robot_x_m, robot_y_m, robot_heading_rad)

        x = y = rot = 0.0
        success = False

        if current_key == "1":
            algaes: List[Algae] = visible_game_pieces.get_algae()
            best_algae = autoalgae.compute_best_algae(algaes)
            if best_algae and best_algae.x:
                x, y, rot, success = autoalgae.get_algae_navigation_command(
                    best_algae)
                if success:
                    angle = MonoVision.get_angle_to_object_in_degrees(
                        best_algae.x, DisplayConfig.FRAME_WIDTH_PX)
                    Display.draw_angle_line(frame, angle)
                    logger.info(
                        f'[TEST] Algae Nav - X: {x:.2f}, Y: {y:.2f}, ROT: {rot:.2f}')
                else:
                    logger.warning("[TEST] Algae pathfinding failed.")

        elif current_key == "2" and processor_apriltag:
            x, y, rot, success = autoprocessor.get_processor_navigation_command(
                processor_apriltag)
            if success and processor_apriltag.center_x:
                angle_to_processor = MonoVision.get_angle_to_object_in_degrees(
                    processor_apriltag.center_x, DisplayConfig.FRAME_WIDTH_PX)
                Display.draw_angle_line(frame, angle_to_processor)
                logger.info(
                    f'[TEST] Target Movement - X: {x}, Y: {y}, ROT: {rot}')
            else:
                logger.warning("[TEST] Cannot Pathfind to Processor")

        if DebugConfig.TESTING:
            messages.append(current_key)
            messages.append(f'X: {x}, Y: {y}, R: {rot}')
            Display.insert_text_onto_frame(frame, messages)
            messages.clear()
            gui.update(camera_frame, odometry_frame)

        if out:
            out.write(camera_frame)

        if DebugConfig.TESTING and ((cv2.waitKey(1) & 0xFF) == ord('q')):
            break
