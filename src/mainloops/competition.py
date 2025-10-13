
from logging import Logger
from typing import List, Optional

import cv2

from apriltags.apriltags import AprilTagDetection
from camera.monovision import MonoVision
from config import DisplayConfig
from src.display import Display
from navigator.autoalgae import AlgaePickupCommand
from navigator.autocoral import CoralPickupCommand
from navigator.autoprocessor import ProcessorScoringCommand
from navigator.autoreef import ReefScoringCommand
from navigator.trackable_objects import Algae, Coral
from networking.roborio import RoboRio
from vision.processor import Processor

def competition(logger: Logger, frame_processor: Processor, roborio: RoboRio, autoalgae: AlgaePickupCommand, autocoral: CoralPickupCommand, autoreef: ReefScoringCommand, autoprocessor: ProcessorScoringCommand, cap: cv2.VideoCapture, out: cv2.VideoWriter) -> None:
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
            (tag for tag in apriltags if tag.id == processor_id),
            None
        )
        reef_apriltags: List[AprilTagDetection] = [
            tag for tag in apriltags if tag.id in reef_ids
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
                            target_coral.x, DisplayConfig.FRAME_WIDTH_PX)
                        Display.draw_angle_line(frame, angle)
                        logger.info(
                            f'[TELEOP] Aligning to Coral — Angle: {angle:.2f}°')

                # elif reef_apriltags:
                #     target_apriltag = autoreef.compute_best_apriltag(reef_apriltags)
                #     if target_apriltag:
                #         angle = MonoVision.get_angle_to_object_in_degrees(target_apriltag.getCenter().x)
                #         Display.draw_angle_line(frame, angle)
                #         logger.info(f'[TELEOP] Aligning to Processor — Angle: {angle:.2f}°')

        out.write(camera_frame)