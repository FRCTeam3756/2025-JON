import os
import cv2
import logging
from logging import Logger
from typing import Optional, List, Tuple
from apriltags.apriltag_finder import AprilTagFinder
from localization.localization import Localization
from logs.logging_setup import setup_logger

from gui import GUI
from config import AprilTagConfig, CameraConfig, DebugConfig, DisplayConfig
from networking.rio_communication import RoboRio
from camera.monovision import MonoVision
from odometry.odometry import Odometry
from vision.display import Display
from vision.processor import Processor
from navigator.autoalgae import AlgaePickupCommand
from navigator.autocoral import CoralPickupCommand
from navigator.autoreef import ReefScoringCommand
from navigator.autoprocessor import ProcessorScoringCommand
from navigator.trackable_objects import Algae, Coral
from robotpy_apriltag import AprilTagDetection

###############################################################


def init() -> Tuple[Logger, GUI, RoboRio, Odometry, Localization, Processor, AlgaePickupCommand, CoralPickupCommand, ReefScoringCommand, ProcessorScoringCommand, cv2.VideoCapture, Optional[cv2.VideoWriter]]:
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    logger = setup_logger(file_name)

    gui = GUI()
    roborio = RoboRio()
    odometry = Odometry()
    localization = Localization()
    frame_processor = Processor()
    autoalgae = AlgaePickupCommand()
    autocoral = CoralPickupCommand()
    autoreef = ReefScoringCommand()
    autoprocessor = ProcessorScoringCommand()

    cap = cv2.VideoCapture(DisplayConfig.INPUT_PATH)
    if not cap.isOpened():
        logger.error(f"Error opening video: {DisplayConfig.INPUT_PATH}")
        raise RuntimeError("Video stream could not be opened")

    out = None
    if DisplayConfig.SAVE_VIDEO:
        fourcc = cv2.VideoWriter.fourcc(*'XVID')
        out = cv2.VideoWriter(DisplayConfig.OUTPUT_VIDEO_PATH, fourcc, 60.0,
                              (CameraConfig.FRAME_WIDTH_PX, CameraConfig.FRAME_HEIGHT_PX), True)

    logger.info("System initialized successfully.")
    return logger, gui, roborio, odometry, localization, frame_processor, autoalgae, autocoral, autoreef, autoprocessor, cap, out


def testing_mainloop(logger: Logger, gui: GUI, odometry: Odometry, localization: Localization, frame_processor: Processor, autoalgae: AlgaePickupCommand, autocoral: CoralPickupCommand, autoreef: ReefScoringCommand, autoprocessor: ProcessorScoringCommand, cap: cv2.VideoCapture, out: Optional[cv2.VideoWriter]) -> None:
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
            (tag for tag in apriltags if tag.getId() == processor_id),
            None
        )
        reef_apriltags: Optional[List[AprilTagDetection]] = [
            tag for tag in apriltags if tag.getId() in reef_ids
        ]
        closest_apriltag = AprilTagFinder.get_best_apriltag(apriltags)
        if closest_apriltag:
            robot_x_m, robot_y_m, robot_heading_rad = localization.get_world_position(closest_apriltag.getId(), MonoVision.get_distance_to_object_in_mm(AprilTagConfig.APRILTAG_SIZE_CM / 10, abs(closest_apriltag.getCorner(0).x - closest_apriltag.getCorner(3).x)), MonoVision.get_angle_to_object_in_degrees(closest_apriltag.getCenter().x))
        odometry.game_pieces.add(visible_game_pieces)
        logger.debug(f'[DEBUG]: Sending {len(visible_game_pieces.get_all())} objects to odometry')
        odometry_frame = odometry.process_frame(robot_x_m, robot_y_m, robot_heading_rad)

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
                        best_algae.x)
                    Display.draw_angle_line(frame, angle)
                    logger.info(
                        f'[TEST] Algae Nav - X: {x:.2f}, Y: {y:.2f}, ROT: {rot:.2f}')
                    if not DebugConfig.TESTING:
                        data = {
                            "x": x,
                            "y": y,
                            "rot": rot,
                            "success": success
                        }
                        roborio.send_data(data)
                else:
                    logger.warning("[TEST] Algae pathfinding failed.")

        elif current_key == "2" and processor_apriltag:
            x, y, rot, success = autoprocessor.get_processor_navigation_command(
                processor_apriltag)
            if success:
                angle_to_processor = MonoVision.get_angle_to_object_in_degrees(
                    processor_apriltag.getCenter().x)
                Display.draw_angle_line(frame, angle_to_processor)
                logger.info(
                    f'[TEST] Target Movement - X: {x}, Y: {y}, ROT: {rot}')
                if not DebugConfig.TESTING:
                    data = {
                        "x": x,
                        "y": y,
                        "rot": rot,
                        "success": success
                    }
                    roborio.send_data(data)
            else:
                logger.warning("[TEST] Cannot Pathfind to Processor")

        if DisplayConfig.SHOW_VIDEO:
            messages.append(current_key)
            messages.append(f'X: {x}, Y: {y}, R: {rot}')
            Display.insert_text_onto_frame(frame, messages)
            messages.clear()
            gui.update(camera_frame, odometry_frame)

        if DisplayConfig.SAVE_VIDEO and out:
            out.write(camera_frame)

        if DisplayConfig.SHOW_VIDEO and cv2.waitKey(1) & 0xFF == ord('q'):
            break


def competition_mainloop(logger: Logger, gui: GUI, frame_processor: Processor, roborio: RoboRio, autoalgae: AlgaePickupCommand, autocoral: CoralPickupCommand, autoreef: ReefScoringCommand, autoprocessor: ProcessorScoringCommand, cap: cv2.VideoCapture, out: Optional[cv2.VideoWriter]) -> None:
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

        task = roborio.get_data("task")
        processor_id = 3 if roborio.get_data("team_colour") == "red" else 16
        reef_ids = {6, 7, 8, 9, 10, 11} if roborio.get_data(
            "team_colour") == "red" else {17, 18, 19, 20, 21, 22}
        processor_apriltag: Optional[AprilTagDetection] = next(
            (tag for tag in apriltags if tag.getId() == processor_id),
            None
        )
        reef_apriltags: List[AprilTagDetection] = [
            tag for tag in apriltags if tag.getId() in reef_ids
        ]

        x = y = rot = 0.0
        success = False

        match task:
            case "auto":
                if not roborio.get_data("has_algae"):
                    algaes: List[Algae] = visible_game_pieces.get_algae()
                    best_algae = autoalgae.compute_best_algae(algaes)

                    if best_algae:
                        x, y, rot, success = autoalgae.get_algae_navigation_command(
                            best_algae)
                        if success:
                            logger.info(
                                f'[AUTO] Target Movement -  X: {x}, Y: {y}, ROT: {rot}')
                            data = {
                                "x": x,
                                "y": y,
                                "rot": rot,
                                "success": success
                            }
                            roborio.send_data(data)
                        else:
                            logger.warning(
                                "[AUTO] Pathfinding to Algae Failed")
                    else:
                        logger.info("[AUTO] Cannot Find Algae")
                elif roborio.get_data("has_algae") and processor_apriltag:
                    x, y, rot, success = autoprocessor.get_processor_navigation_command(
                        processor_apriltag)
                    if success:
                        logger.info(
                            f'[AUTO] Target Movement -  X: {x}, Y: {y}, ROT: {rot}')
                        data = {
                            "x": x,
                            "y": y,
                            "rot": rot,
                            "success": success
                        }
                        roborio.send_data(data)
                    else:
                        logger.warning(
                            "[AUTO] Pathfinding to Processor Failed")
                else:
                    logger.info("[AUTO] Cannot Find Processor")

            case "teleop":
                corals: List[Coral] = visible_game_pieces.get_coral()
                if corals:
                    target_coral = autocoral.compute_best_coral(corals)
                    if target_coral and target_coral.x:
                        angle = MonoVision.get_angle_to_object_in_degrees(
                            target_coral.x)
                        Display.draw_angle_line(frame, angle)
                        logger.info(
                            f'[TELEOP] Aligning to Coral — Angle: {angle:.2f}°')

                # elif reef_apriltags:
                #     target_apriltag = autoreef.compute_best_apriltag(reef_apriltags)
                #     if target_apriltag:
                #         angle = MonoVision.get_angle_to_object_in_degrees(target_apriltag.getCenter().x)
                #         Display.draw_angle_line(frame, angle)
                #         logger.info(f'[TELEOP] Aligning to Processor — Angle: {angle:.2f}°')

        if DisplayConfig.SHOW_VIDEO:
            messages.append(task)
            messages.append(f'X: {x}, Y: {y}, R: {rot}')
            Display.insert_text_onto_frame(frame, messages)
            messages = []
            Display.show_frame(DisplayConfig.WINDOW_TITLE, camera_frame)

        if DisplayConfig.SAVE_VIDEO and out:
            out.write(camera_frame)

        if DisplayConfig.SHOW_VIDEO and cv2.waitKey(1) & 0xFF == ord('q'):
            break

###############################################################


if __name__ == "__main__":
    logger, gui, roborio, odometry, localization, frame_processor, autoalgae, autocoral, autoreef, autoprocessor, cap, out = init()
    try:
        if DebugConfig.TESTING:
            testing_mainloop(logger, gui, odometry, localization, frame_processor, autoalgae,
                             autocoral, autoreef, autoprocessor, cap, out)
        else:
            competition_mainloop(logger, gui, frame_processor, roborio,
                                 autoalgae, autocoral, autoreef, autoprocessor, cap, out)
    finally:
        cap.release()
        if DisplayConfig.SAVE_VIDEO and out:
            out.release()
        gui.close()
        cv2.destroyAllWindows()
        logging.shutdown()
