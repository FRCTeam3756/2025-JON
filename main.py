import os
import cv2
import logging
from typing import Any, Optional, List

from networking.rio_communication import RoboRio
from vision_tracking.video_display import VideoDisplay
from vision_tracking.video_processor import FrameProcessor
from decision_engine.decision_matrix import DecisionMatrix
from decision_engine.trackable_objects import Algae, Cage, CagePole, Chain, Coral, Robot

###############################################################

script_name: str = os.path.splitext(os.path.basename(__file__))[0]
log_file: str = os.path.join("logs", f"{script_name}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Config:
    """Configuration settings for video processing."""
    # Display
    WINDOW_TITLE: str = 'Output Video'
    SHOW_VIDEO: bool = True
    SAVE_VIDEO: bool = True
    FRAME_WIDTH: int = 640
    FRAME_HEIGHT: int = 640
    LABEL_COLOURS: dict[str, list[int]] = {
        "0": [85, 186, 151],    # Algae
        "1": [0, 0, 0],         # Cage
        "2": [0, 0, 255],       # Cage Pole
        "3": [149, 149, 149],   # Chain
        "4": [255, 255, 255],   # Coral
        "5": [121, 217, 255],   # Robot
    }
    
    # Image Transformations
    ROTATE_IMAGE: bool = False
    FLIP_IMAGE_HORIZONTALLY: bool = False
    FLIP_IMAGE_VERTICALLY: bool = False

    # AI
    COVERAGE_THRESHOLD: float = 0.4
    CONFIDENCE_THRESHOLD: float = 0.7

    # Paths
    INPUT_VIDEO_PATH: Any = "test/input/video2.mp4" #"http://limelight.local:5800" #0 #
    OUTPUT_VIDEO_PATH: str = 'test/output/output.mp4'
    WEIGHTS_LOCATION: str = 'vision_tracking/weights/best.onnx'

    # Logging
    FPS_LOGGING_RATE: int = 200

def main() -> None:
    roborio: RoboRio = RoboRio()
    decision_matrix: DecisionMatrix = DecisionMatrix()
    processor: FrameProcessor = FrameProcessor(Config)
    cap: cv2.VideoCapture = cv2.VideoCapture(Config.INPUT_VIDEO_PATH)
    
    if not cap.isOpened():
        logging.error(f"Error opening video: {Config.INPUT_VIDEO_PATH}")
        return
    
    out: Optional[cv2.VideoWriter] = None
    if Config.SAVE_VIDEO:
        fourcc: int = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(Config.OUTPUT_VIDEO_PATH, fourcc, 30.0, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
    
    try:
        print("Made connection to cap")
        while cap.isOpened():
            roborio.get_data("placeholder")

            ret, frame = cap.read()
            if not ret:
                logging.info("End of video stream.")
                break

            frame = processor.transform_frame(frame)
            
            processed_frame, game_pieces = processor.process_frame(frame)
            
            processor.calculate_frame_rate()

            algae: List[Algae] = game_pieces.get(Algae, [])

            best_algae: Optional[Algae] = decision_matrix.compute_best_game_piece(*algae)

            if best_algae:
                roborio.send_data((best_algae.distance, best_algae.angle))
                
            if Config.SHOW_VIDEO:
                if best_algae:
                    VideoDisplay.draw_angle_line(frame, best_algae.angle)
                VideoDisplay.show_frame(Config.WINDOW_TITLE, processed_frame)
                
            if Config.SAVE_VIDEO and out:
                out.write(processed_frame)

            if Config.SHOW_VIDEO and cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if Config.SAVE_VIDEO and out:
            out.release()
            logging.info("Video file closed properly.")
        cv2.destroyAllWindows()

###############################################################

if __name__ == "__main__":
    main()