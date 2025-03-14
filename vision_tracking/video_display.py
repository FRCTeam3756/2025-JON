import cv2
import math
import logging

class VideoDisplay:
    @staticmethod
    def show_frame(frame):
        cv2.imshow('Video', frame)

    @staticmethod
    def annotate_frame(frame, boxes, class_ids, label_colours):
        """Annotate the frame with bounding boxes and labels."""
        for i, (x1, y1, x2, y2) in enumerate(boxes):
            color = label_colours.get(str(class_ids[i]), (255, 255, 255))
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        return frame

    @staticmethod
    def draw_angle_line(frame, angle):
        """Draws a line at a given angle from the bottom center of the screen."""
        height, width = frame.shape[:2]
        start_point = (width // 2, height - 1)
        length = 100
        end_x = int(start_point[0] + length * math.sin(math.radians(angle)))
        end_y = int(start_point[1] - length * math.cos(math.radians(angle)))
        
        if 0 <= end_x < width and 0 <= end_y < height:
            cv2.line(frame, start_point, (end_x, end_y), (0, 155, 255), 2)
        else:
            logging.warning(f"Line endpoint out of bounds: ({end_x}, {end_y}) for angle {angle}")