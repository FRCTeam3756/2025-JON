import os
import cv2
import logging
import keyboard
from typing import Optional, List
from logs.logging_setup import setup_logger

from config import *
from networking.rio_communication import RoboRio
from camera_calculations.mono_video import MonoVision
from vision_tracking.video_display import VideoDisplay
from vision_tracking.video_processor import FrameProcessor
from decision_engine.autoalgae import AlgaePickupCommand
from decision_engine.autocoral import CoralPickupCommand
from decision_engine.autoreef import ReefScoringCommand
from decision_engine.autoprocessor import ProcessorScoringCommand
from decision_engine.trackable_objects import *
from robotpy_apriltag import AprilTagDetection

###############################################################

def main() -> None:
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    logger = setup_logger(file_name)

    roborio = RoboRio()
    autoalgae = AlgaePickupCommand()
    autocoral = CoralPickupCommand()
    autoreef = ReefScoringCommand()
    autoprocessor = ProcessorScoringCommand()
    frame_processor = FrameProcessor()

    cap = cv2.VideoCapture(DisplayConfig.INPUT_VIDEO_PATH)
    if not cap.isOpened():
        logger.error(f"Error opening video: {DisplayConfig.INPUT_VIDEO_PATH}")
        return
    
    out: Optional[cv2.VideoWriter] = None
    if DisplayConfig.SAVE_VIDEO:
        fourcc: int = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(DisplayConfig.OUTPUT_VIDEO_PATH, fourcc, 60.0, 
                              (CameraConfig.FRAME_WIDTH, CameraConfig.FRAME_HEIGHT), True)

    if DebugConfig.TESTING:
        current_key: str = None

    messages: List = []
    
    try:
        logger.info("Video stream opened successfully.")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logger.info("End of video stream.")
                break

            frame = frame_processor.transform_frame(frame)
            processed_frame, game_pieces, apriltags = frame_processor.process_frame(frame)
            frame_processor.calculate_frame_rate()
            
            task = roborio.get_data("task") if not DebugConfig.TESTING else DebugConfig.DEFAULT_TASK
            if DebugConfig.TESTING:
                for key in DebugConfig.TASK_KEYS:
                    if keyboard.is_pressed(key):
                        current_key = key
                        break
                if not current_key:
                    current_key = DebugConfig.DEFAULT_KEY

            processor_id = 3 if roborio.get_data("team_colour") == "red" else 16
            processor_apriltag: Optional[AprilTagDetection] = next(
                (tag for tag in apriltags if tag.getId() == processor_id), 
                None
            )
            reef_ids = {6, 7, 8, 9, 10, 11} if roborio.get_data("team_colour") == "red" else {17, 18, 19, 20, 21, 22}
            reef_apriltags: List[AprilTagDetection] = [
                tag for tag in apriltags if tag.getId() in reef_ids
            ]
            
            x = y = rot = 0.0
            success = False

            match task:
                case "auto":
                    if not roborio.get_data("has_algae"):
                        algaes: List[Algae] = game_pieces.get(Algae, [])
                        best_algae = autoalgae.compute_best_algae(algaes)
                       
                        if best_algae: 
                            x, y, rot, success = autoalgae.get_algae_navigation_command(best_algae)
                            if success:
                                logger.info(f'[AUTO] Target Movement -  X: {x}, Y: {y}, ROT: {rot}')
                                if not DebugConfig.TESTING:
                                    roborio.send_data(x, y, rot, success)
                            else:
                                logger.warning("[AUTO] Cannot Pathfind to Algae")
                    elif roborio.get_data("has_algae") and processor_apriltag:
                        x, y, rot, success = autoprocessor.get_processor_navigation_command(processor_apriltag)
                        if success:
                            logger.info(f'[AUTO] Target Movement -  X: {x}, Y: {y}, ROT: {rot}')
                            if not DebugConfig.TESTING:
                                roborio.send_data(x, y, rot, success)
                        else:
                            logger.warning("[AUTO] Cannot Pathfind to Processor")

                case "teleop":
                    corals: List[Coral] = game_pieces.get(Coral, [])
                    if corals:
                        target_coral = autocoral.compute_best_coral(corals)
                        if target_coral:
                            angle = MonoVision.get_angle_to_object_in_degrees(target_coral.x)
                            VideoDisplay.draw_angle_line(frame, angle)
                            logger.info(f'[TELEOP] Aligning to Coral — Angle: {angle:.2f}°')

                    elif reef_apriltags:
                        target_apriltag = autoreef.compute_best_apriltag(reef_apriltags)
                        if target_apriltag:
                            angle = MonoVision.get_angle_to_object_in_degrees(target_apriltag.getCenter().x)
                            VideoDisplay.draw_angle_line(frame, angle)
                            logger.info(f'[TELEOP] Aligning to Processor — Angle: {angle:.2f}°')

                case "test":
                    if current_key == "1":
                        algaes: List[Algae] = game_pieces.get(Algae, [])
                        best_algae = autoalgae.compute_best_algae(algaes)
                        if best_algae:
                            x, y, rot, success = autoalgae.get_algae_navigation_command(best_algae)
                            if success:
                                angle = MonoVision.get_angle_to_object_in_degrees(best_algae.x)
                                VideoDisplay.draw_angle_line(frame, angle)
                                logger.info(f'[TEST] Algae Nav → X: {x:.2f}, Y: {y:.2f}, ROT: {rot:.2f}')
                                if not DebugConfig.TESTING:
                                    roborio.send_data(x, y, rot, success)
                            else:
                                logger.warning("[TEST] Algae pathfinding failed.")
                    elif current_key == "2" and processor_apriltag:
                        x, y, rot, success = autoprocessor.get_processor_navigation_command(processor_apriltag) 
                        if success:
                            angle_to_processor = MonoVision.get_angle_to_object_in_degrees(processor_apriltag.getCenter().x)
                            VideoDisplay.draw_angle_line(frame, angle_to_processor)
                            logger.info(f'[TEST] Target Movement -  X: {x}, Y: {y}, ROT: {rot}')
                            if not DebugConfig.TESTING:
                                roborio.send_data(x, y, rot, success)
                        else:
                            logger.warning("[TEST] Cannot Pathfind to Processor")
                
            if DisplayConfig.SHOW_VIDEO:
                messages.append(current_key)
                messages.append(task)
                messages.append(f'X: {x}, Y: {y}, R: {rot}')
                VideoDisplay.insert_text_onto_frame(frame, messages)
                messages = []
                VideoDisplay.show_frame(DisplayConfig.WINDOW_TITLE, processed_frame)
            
            if DisplayConfig.SAVE_VIDEO and out:
                out.write(processed_frame)

            if DisplayConfig.SHOW_VIDEO and cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if DisplayConfig.SAVE_VIDEO and out:
            out.release()
            logger.info("Video file closed properly.")
        cv2.destroyAllWindows()
        logging.shutdown()

###############################################################

if __name__ == "__main__":
    main()