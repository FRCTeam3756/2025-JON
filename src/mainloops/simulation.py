import time
from logging import Logger
from typing import List, Optional

import cv2

from config import DebugConfig, DisplayConfig
from src.apriltags.apriltags import AprilTagDetection, AprilTagFinder
from src.camera.monovision import MonoVision
from src.display import Display
from src.gui import GUI
from src.localization.localization import Localization
from src.navigator.autoalgae import AlgaePickupCommand
from src.navigator.autocoral import CoralPickupCommand
from src.navigator.autoprocessor import ProcessorScoringCommand
from src.navigator.autoreef import ReefScoringCommand
from src.navigator.trackable_objects import Algae
from src.odometry.odometry import Odometry
from src.vision.processor import Processor


def simulation(logger: Logger, gui: GUI, odometry: Odometry, localization: Localization, frame_processor: Processor, autoalgae: AlgaePickupCommand, autocoral: CoralPickupCommand, autoreef: ReefScoringCommand, autoprocessor: ProcessorScoringCommand, cap: cv2.VideoCapture, out: cv2.VideoWriter) -> None:
    logger.info("Running in TESTING mode.")

    # Starting Location
    robot_x_m, robot_y_m, robot_heading_rad = 8, 2, 2.9
    messages: List = []

    while cap.isOpened():
        frame_start = time.perf_counter()

        t0 = time.perf_counter()
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video stream.")
            break
        t1 = time.perf_counter()

        frame = frame_processor.transform_frame(frame)
        t2 = time.perf_counter()
        camera_frame, visible_game_pieces, apriltags = frame_processor.process_frame(
            frame)
        t3 = time.perf_counter()

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
        if closest_apriltag and closest_apriltag.relative_distance_m and closest_apriltag.relative_angle_deg:
            robot_x_m, robot_y_m, robot_heading_rad = localization.get_world_position(closest_apriltag.id, closest_apriltag.relative_distance_m, closest_apriltag.relative_angle_deg)
        t4 = time.perf_counter()

        odometry.game_pieces.add(visible_game_pieces)
        logger.debug(
            f'[DEBUG]: Sending {len(visible_game_pieces.get_all())} objects to odometry')
        odometry_frame = odometry.process_frame(
            robot_x_m, robot_y_m, robot_heading_rad)
        t5 = time.perf_counter()

        x = y = rot = 0.0
        success = False

        if DebugConfig.TASK == 0:
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

        elif DebugConfig.TASK == 1:
            if processor_apriltag:
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
            else:
                logger.warning("[TEST] Processor not Found")
        t6 = time.perf_counter()

        messages.append(f'X: {x}, Y: {y}, R: {rot}')
        Display.insert_text_onto_frame(camera_frame, messages)
        messages.clear()
        gui.update(camera_frame, odometry_frame)
        out.write(camera_frame)
        t7 = time.perf_counter()
        
        frame_end = time.perf_counter()
        total_time_ms = (frame_end - frame_start) * 1000

        # 🧾 LOGGING DELAYS
        logger.debug((
            f"[TIMING] Frame Capture: {(t1 - t0)*1000:.2f} ms | "
            f"Transform: {(t2 - t1)*1000:.2f} ms | "
            f"Process Frame: {(t3 - t2)*1000:.2f} ms | "
            f"Localization: {(t4 - t3)*1000:.2f} ms | "
            f"Odometry: {(t5 - t4)*1000:.2f} ms | "
            f"Navigation: {(t6 - t5)*1000:.2f} ms | "
            f"Display/GUI: {(t7 - t6)*1000:.2f} ms | "
            f"TOTAL: {total_time_ms:.2f} ms"
        ))

        if (cv2.waitKey(1) & 0xFF) == ord('q'):
            break
