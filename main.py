import os
import cv2
import logging
from typing import Tuple
from logging import Logger
from dataclasses import dataclass

from logs.logging_setup import setup_logger
from config import DebugConfig, DisplayConfig

from src.mainloops.simulation import simulation
from src.mainloops.competition import competition
from src.localization.localization import Localization
from src.gui import GUI
from src.networking.roborio import RoboRio
from src.odometry.odometry import Odometry
from src.vision.processor import Processor
from src.navigator.autoalgae import AlgaePickupCommand
from src.navigator.autocoral import CoralPickupCommand
from src.navigator.autoreef import ReefScoringCommand
from src.navigator.autoprocessor import ProcessorScoringCommand

###############################################################

@dataclass
class RobotSystems:
    logger: Logger
    gui: GUI
    roborio: RoboRio
    odometry: Odometry
    localization: Localization
    processor: Processor
    autoalgae: AlgaePickupCommand
    autocoral: CoralPickupCommand
    autoreef: ReefScoringCommand
    autoprocessor: ProcessorScoringCommand
    cap: cv2.VideoCapture
    out: cv2.VideoWriter

###############################################################

def setup_video_io() -> Tuple[cv2.VideoCapture, cv2.VideoWriter]:
    cap = cv2.VideoCapture(DisplayConfig.INPUT_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Error opening video: {DisplayConfig.INPUT_PATH}")

    out = None
    fourcc = cv2.VideoWriter.fourcc(*'XVID')
    out = cv2.VideoWriter(
        DisplayConfig.OUTPUT_VIDEO_PATH,
        fourcc,
        30.0,
        (DisplayConfig.FRAME_WIDTH_PX, DisplayConfig.FRAME_HEIGHT_PX),
        True
    )

    return cap, out

def init_robot_systems() -> RobotSystems:
    file_name = os.path.splitext(os.path.basename(__file__))[0]
    logger = setup_logger(file_name)

    try:
        gui = GUI()
        roborio = RoboRio()
        odometry = Odometry()
        localization = Localization()
        processor = Processor()
        autoalgae = AlgaePickupCommand()
        autocoral = CoralPickupCommand()
        autoreef = ReefScoringCommand()
        autoprocessor = ProcessorScoringCommand()
        cap, out = setup_video_io()
    except Exception as e:
        logger.exception("Failed to initialize robot systems")
        raise e

    logger.info("System initialized successfully.")
    return RobotSystems(logger, gui, roborio, odometry, localization,
                        processor, autoalgae, autocoral, autoreef, autoprocessor, cap, out)

###############################################################


if __name__ == "__main__":
    systems = init_robot_systems()

    try:
        if DebugConfig.TESTING:
            simulation(
                systems.logger, systems.gui, systems.odometry, systems.localization,
                systems.processor, systems.autoalgae, systems.autocoral,
                systems.autoreef, systems.autoprocessor, systems.cap, systems.out
            )
        else:
            competition(
                systems.logger, systems.processor, systems.roborio,
                systems.autoalgae, systems.autocoral, systems.autoreef,
                systems.autoprocessor, systems.cap, systems.out
            )
    finally:
        systems.cap.release()
        systems.out.release()
        systems.gui.close()
        cv2.destroyAllWindows()
        logging.shutdown()
