import os
import cv2
import logging

from vision_tracking.video_display import VideoDisplay
from vision_tracking.video_processor import FrameProcessor
from networking.rio_communication import post_to_network_tables

###############################################################

script_name = os.path.splitext(os.path.basename(__file__))[0]

log_file = os.path.join("logs", f"{script_name}.log")

logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

class Config:
    """Configuration settings for video processing."""
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 640
    
    ROTATE_IMAGE: bool = True
    FLIP_IMAGE_HORIZONTALLY: bool = True
    FLIP_IMAGE_VERTICALLY: bool = False

    MAXIMUM_FRAME_RATE: int = 200
    COVERAGE_THRESHOLD: float = 0.4
    CONFIDENCE_THRESHOLD: float = 0.7
    DISPLAY: bool = True
    SAVE_VIDEO: bool = True
    VIDEO_PATH: str = "video.mp4" #"http://limelight.local:5800" #0 #
    WEIGHTS_LOCATION: str = 'vision_tracking/runs/train/weights/best.onnx'
    LABEL_COLOURS: dict[str, list[int]] = {
        "0": [0, 155, 255],
        "1": [0, 0, 255]
    }

def main():
    processor = FrameProcessor(Config())
    cap = cv2.VideoCapture(Config.VIDEO_PATH)
    
    if not cap.isOpened():
        logging.error(f"Error opening video: {Config.VIDEO_PATH}")
    
    if Config.SAVE_VIDEO:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (Config.FRAME_WIDTH, Config.FRAME_HEIGHT))
    
    try:
        print("Made connection to cap")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.info("End of video stream.")
                break

            frame = processor.fix_frame(frame)
            
            processed_frame, note = processor.process_frame(frame)
            if note:
                post_to_network_tables((note.distance, note.angle))
                
            if Config.DISPLAY:
                if note:
                    VideoDisplay.draw_angle_line(frame, note.angle)
                VideoDisplay.show_frame(processed_frame)
                
            if Config.SAVE_VIDEO:
                out.write(processed_frame)
                
            processor.calculate_frame_rate()

            if Config.DISPLAY & cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        if Config.SAVE_VIDEO:
            out.release()
            logging.info("Video file closed properly.")
        cv2.destroyAllWindows()

###############################################################

if __name__ == "__main__":
    main()