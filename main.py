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
    ROTATE_IMAGE: bool = False
    FLIP_IMAGE_HORIZONTALLY: bool = True
    FLIP_IMAGE_VERTICALLY: bool = False

    FRAME_RATE_REFRESH_RATE: int = 200
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
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        out = cv2.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))
    
    try:
        print("Made connection to cap. Look for reaction on DS")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.warning("End of video stream.")
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
                out.write(cv2.cvtColor(processed_frame, cv2.COLOR_RGB2BGR))
                
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