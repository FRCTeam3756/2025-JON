import os
import cv2
import logging
from typing import Optional, List

from config import DisplayConfig
from networking.rio_communication import RoboRio
from vision_tracking.video_display import VideoDisplay
from vision_tracking.video_processor import FrameProcessor
from decision_engine.autoalgae import DecisionMatrix
from decision_engine.trackable_objects import Algae, Cage, CagePole, Chain, Coral, Robot

###############################################################

script_name: str = os.path.splitext(os.path.basename(__file__))[0]
log_file: str = os.path.join("logs", f"{script_name}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def main() -> None:
    roborio: RoboRio = RoboRio()
    decision_matrix: DecisionMatrix = DecisionMatrix()
    processor: FrameProcessor = FrameProcessor()
    cap: cv2.VideoCapture = cv2.VideoCapture(DisplayConfig.INPUT_VIDEO_PATH)
    
    if not cap.isOpened():
        logging.error(f"Error opening video: {DisplayConfig.INPUT_VIDEO_PATH}")
        return
    
    out: Optional[cv2.VideoWriter] = None
    if DisplayConfig.SAVE_VIDEO:
        fourcc: int = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(DisplayConfig.OUTPUT_VIDEO_PATH, fourcc, 30.0, (DisplayConfig.FRAME_WIDTH, DisplayConfig.FRAME_HEIGHT))
    
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
                
            if DisplayConfig.SHOW_VIDEO:
                if best_algae:
                    VideoDisplay.draw_angle_line(frame, best_algae.angle)
                VideoDisplay.show_frame(DisplayConfig.WINDOW_TITLE, processed_frame)
                
            if DisplayConfig.SAVE_VIDEO and out:
                out.write(processed_frame)

            if DisplayConfig.SHOW_VIDEO and cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if DisplayConfig.SAVE_VIDEO and out:
            out.release()
            logging.info("Video file closed properly.")
        cv2.destroyAllWindows()

###############################################################

if __name__ == "__main__":
    main()