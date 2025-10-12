import os
import cv2
import math
import numpy as np
from typing import Tuple

from config import CameraConfig, FieldConfig, RamFernoRobotConfig
from logs.logging_setup import setup_logger
from navigator.trackable_objects import Algae, Coral, Cage, GamePieces, Robot


class Odometry:
    PIXELS_PER_METER = 70
    MARGIN_PX = 20
    INCH_TO_M = 0.0254

    ROBOT_COLOR = (0, 125, 255)
    APRILTAG_COLOR = (50, 50, 50)
    DETECTION_COLOR_MAP = {
        Algae: (0, 255, 0),
        Cage: (255, 0, 0),
        Coral: (0, 255, 255),
        Robot: (255, 0, 255)
    }
    WINDOW_NAME = "FRC 2025 Odometry Demo"

    APRILTAG_POSITIONS = {
        1: (656.98, 24.73, 126),
        2: (656.98, 291.90, 234),
        3: (452.40, 316.21, 270),
        4: (365.20, 241.44, 0),
        5: (365.20, 75.19, 0),
        6: (530.49, 129.97, 300),
        7: (546.87, 158.30, 0),
        8: (530.49, 186.63, 60),
        9: (497.77, 186.63, 120),
        10: (481.39, 158.30, 180),
        11: (497.77, 129.97, 240),
        12: (33.91, 24.73, 54),
        13: (33.91, 291.90, 306),
        14: (325.68, 241.44, 180),
        15: (325.68, 75.19, 180),
        16: (238.49, 0.42, 90),
        17: (160.39, 129.97, 240),
        18: (144.00, 158.30, 180),
        19: (160.39, 186.63, 120),
        20: (193.10, 186.63, 60),
        21: (209.49, 158.30, 0),
        22: (193.10, 129.97, 300),
    }

    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        self.logger = setup_logger(file_name)
        self.logger.info("Odometry logger initialized.")

        self.fps = 30
        self.target_x = FieldConfig.FIELD_WIDTH_M * 0.85
        self.target_y = FieldConfig.FIELD_HEIGHT_M * 0.75
        self.game_pieces = GamePieces()

        self.APRILTAG_POSITIONS = {
            k: (x * self.INCH_TO_M, y * self.INCH_TO_M, r)
            for k, (x, y, r) in self.APRILTAG_POSITIONS.items()
        }

        self.IMG_W = (
            int(FieldConfig.FIELD_WIDTH_M *
                self.PIXELS_PER_METER) + 2 * self.MARGIN_PX
        )
        self.IMG_H = (
            int(FieldConfig.FIELD_HEIGHT_M *
                self.PIXELS_PER_METER) + 2 * self.MARGIN_PX
        )

        self.TAG_SIZE_M = 0.35

    def camera_to_canvas(self, x_m: float, y_m: float) -> Tuple[int, int]:
        px = int(self.MARGIN_PX + x_m * self.PIXELS_PER_METER)
        py = int(self.IMG_H - (self.MARGIN_PX + y_m * self.PIXELS_PER_METER))
        return px, py

    def draw_field(self, canvas: np.ndarray) -> None:
        self.logger.debug("Drawing field.")
        top_left = (self.MARGIN_PX, self.MARGIN_PX)
        bottom_right = (self.IMG_W - self.MARGIN_PX,
                        self.IMG_H - self.MARGIN_PX)
        canvas[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]] = (
            50,
            120,
            50,
        )
        cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), 2)
        canvas_x_px = int(self.MARGIN_PX + (FieldConfig.FIELD_WIDTH_M / 2)
                          * self.PIXELS_PER_METER)
        cv2.line(
            canvas,
            (canvas_x_px, self.MARGIN_PX),
            (canvas_x_px, self.IMG_H - self.MARGIN_PX),
            (200, 200, 200),
            1,
        )

    def draw_apriltags(self, canvas: np.ndarray) -> None:
        self.logger.debug("Drawing AprilTags on field.")
        half_px = int((self.TAG_SIZE_M * self.PIXELS_PER_METER) / 2)
        for (tag_id, (x_m, y_m, rot_deg)) in (
            self.APRILTAG_POSITIONS.items()
        ):
            px, py = self.camera_to_canvas(x_m, y_m)
            cv2.rectangle(
                canvas,
                (px - half_px, py - half_px),
                (px + half_px, py + half_px),
                self.APRILTAG_COLOR,
                -1,
            )
            cv2.rectangle(
                canvas,
                (px - half_px, py - half_px),
                (px + half_px, py + half_px),
                (0, 0, 0),
                2,
            )
            cv2.putText(
                canvas,
                str(tag_id),
                (px - half_px + 4, py + 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
            rad = math.radians(rot_deg)
            fx = int(px + math.cos(rad) * half_px * 1.3)
            fy = int(py - math.sin(rad) * half_px * 1.3)
            cv2.line(canvas, (px, py), (fx, fy), (0, 0, 0), 2)

    def draw_ramferno(self, canvas: np.ndarray, robot_x_m: float, robot_y_m: float, robot_heading_rad: float) -> None:
        self.logger.debug(f"Drawing robot at ({robot_x_m:.2f}, {robot_y_m:.2f}) with heading {math.degrees(robot_heading_rad):.1f}°")

        robot_field_x, robot_field_y = self.camera_to_canvas(
            robot_x_m, robot_y_m)

        robot_half_width_px = (RamFernoRobotConfig.ROBOT_WIDTH_M * self.PIXELS_PER_METER) / 2
        robot_half_length_px = (RamFernoRobotConfig.ROBOT_LENGTH_M * self.PIXELS_PER_METER) / 2
        corners = [
            (-robot_half_length_px, -robot_half_width_px), # back left
            (robot_half_length_px, -robot_half_width_px),  # front left
            (robot_half_length_px, robot_half_width_px),   # front right
            (-robot_half_length_px, robot_half_width_px),  # back right
        ]

        def rot(lx, ly) -> Tuple[int, int]:
            rx = lx * math.cos(robot_heading_rad) - ly * \
                math.sin(robot_heading_rad)
            ry = lx * math.sin(robot_heading_rad) + ly * \
                math.cos(robot_heading_rad)
            return int(robot_field_x + rx), int(robot_field_y - ry)

        pts = np.array([rot(x, y) for x, y in corners], np.int32)
        cv2.fillConvexPoly(canvas, pts, self.ROBOT_COLOR)
        cv2.polylines(canvas, [pts], True, (0, 0, 150), 2)

        front_local = [(robot_half_length_px, -robot_half_width_px),
                       (robot_half_length_px, robot_half_width_px)]
        front_pts = np.array([rot(x, y) for x, y in front_local], np.int32)
        cv2.line(canvas, tuple(front_pts[0]),
                 tuple(front_pts[1]), (0, 0, 255), 3)

    def draw_target_position(self, canvas: np.ndarray, robot_x_m: float, robot_y_m: float) -> None:
        robot_canvas_x, robot_canvas_y = self.camera_to_canvas(
            robot_x_m, robot_y_m)
        tx, ty = self.camera_to_canvas(self.target_x, self.target_y)
        cv2.circle(canvas, (tx, ty), 6, (0, 255, 0), -1)
        cv2.line(canvas, (robot_canvas_x, robot_canvas_y),
                 (tx, ty), (0, 200, 0), 1)
        dist = math.hypot(self.target_x - robot_x_m, self.target_y - robot_y_m)
        midx, midy = (robot_canvas_x + tx) // 2, (robot_canvas_y + ty) // 2
        cv2.putText(
            canvas,
            f"{dist:.2f}m",
            (midx, midy - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

    def draw_detections(self, canvas: np.ndarray, robot_x_m, robot_y_m, robot_heading_rad):
        """Draw all additional tracked objects."""
        for cls, objs in list(self.game_pieces._data.items()):
            for obj in list(objs):
                if obj.relative_distance_mm is None or obj.relative_angle_deg is None:
                    self.logger.warning(f"Skipping {cls.__name__}: Missing distance or angle data.") 
                    continue
                
                obj_x_m, obj_y_m = self.object_world_coords(
                    obj.relative_distance_mm, obj.relative_angle_deg, robot_x_m, robot_y_m, robot_heading_rad
                )

                px, py = self.camera_to_canvas(obj_x_m, obj_y_m)

                self.logger.debug(f"Drawing {cls.__name__} at ({px}, {py}) world=({obj_x_m:.2f},{obj_y_m:.2f})m")

                color = self.DETECTION_COLOR_MAP.get(cls, (200, 200, 200))
                cv2.circle(canvas, (px, py), 6, color, -1)
                cv2.putText(
                    canvas,
                    cls.__name__,
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color,
                    1,
                    cv2.LINE_AA,
                )

    def draw_vision_cone(self, canvas: np.ndarray, robot_x_m, robot_y_m, robot_heading_rad) -> None:
        """Draws a semi-transparent grey cone representing robot vision."""
        left_angle = robot_heading_rad - \
            math.radians(CameraConfig.HORIZONTAL_FOV_DEG) / 2
        right_angle = robot_heading_rad + \
            math.radians(CameraConfig.HORIZONTAL_FOV_DEG) / 2

        robot_canvas_x, robot_canvas_y = self.camera_to_canvas(
            robot_x_m, robot_y_m)
        end_left = self.camera_to_canvas(
            robot_x_m + CameraConfig.VISION_RANGE_M * math.cos(left_angle),
            robot_y_m + CameraConfig.VISION_RANGE_M * math.sin(left_angle)
        )
        end_right = self.camera_to_canvas(
            robot_x_m + CameraConfig.VISION_RANGE_M * math.cos(right_angle),
            robot_y_m + CameraConfig.VISION_RANGE_M * math.sin(right_angle)
        )
        cone_pts = np.array(
            [[robot_canvas_x, robot_canvas_y], end_left, end_right], np.int32)

        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, cone_pts, (100, 100, 100))
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

    def object_world_coords(self, relative_distance_mm: float, relative_angle_deg: float, robot_x_m: float, robot_y_m: float, robot_heading_rad: float) -> Tuple[float, float]:
        dist_m = relative_distance_mm * 0.001
        rel_angle_rad = math.radians(relative_angle_deg)
        abs_angle_rad = robot_heading_rad + rel_angle_rad
        obj_x_m = robot_x_m + dist_m * math.cos(abs_angle_rad)
        obj_y_m = robot_y_m + dist_m * math.sin(abs_angle_rad)
        return obj_x_m, obj_y_m

    def object_visible(self, robot_x_m: float, robot_y_m: float, robot_heading_rad: float, obj_x_m: float, obj_y_m: float) -> bool:
        """Checks if object is within robot's vision cone."""
        distance_x = obj_x_m - robot_x_m
        distance_y = obj_y_m - robot_y_m
        distance_total = math.hypot(distance_x, distance_y)
        if distance_total > CameraConfig.VISION_RANGE_M:
            return False

        angle_to_obj = math.atan2(distance_y, distance_x)
        angle_diff = (angle_to_obj - robot_heading_rad +
                      math.pi) % (2 * math.pi) - math.pi

        return abs(angle_diff) <= math.radians(CameraConfig.HORIZONTAL_FOV_DEG) / 2

    def update_past_detections(self, robot_x_m, robot_y_m, robot_heading_rad) -> None:
        self.logger.debug("Updating visibility of past detections.")
        removed = 0
        for cls, objs in list(self.game_pieces._data.items()):
            for obj in list(objs):
                if obj.relative_distance_mm is None or obj.relative_angle_deg is None:
                    continue

                obj_x_m, obj_y_m = self.object_world_coords(
                    obj.relative_distance_mm, obj.relative_angle_deg, robot_x_m, robot_y_m, robot_heading_rad
                )

                if self.object_visible(robot_x_m, robot_y_m, robot_heading_rad, obj_x_m, obj_y_m):
                    self.game_pieces._data[cls].remove(obj)
                    removed += 1
        if removed > 0:
            self.logger.info(f"Removed {removed} objects that stayed within the vision cone.")

    def render_frame(self, robot_x_m: float, robot_y_m: float, robot_heading_rad: float) -> np.ndarray:
        canvas = np.zeros((self.IMG_H, self.IMG_W, 3), np.uint8)

        self.draw_field(canvas)
        self.draw_apriltags(canvas)
        self.draw_ramferno(canvas, robot_x_m, robot_y_m, robot_heading_rad)
        self.draw_vision_cone(canvas, robot_x_m, robot_y_m, robot_heading_rad)
        self.draw_detections(canvas, robot_x_m, robot_y_m, robot_heading_rad)

        return canvas

    def process_frame(self, robot_x_m: float, robot_y_m: float, robot_heading_rad: float) -> np.ndarray:
        frame = self.render_frame(robot_x_m, robot_y_m, robot_heading_rad)
        self.update_past_detections(robot_x_m, robot_y_m, robot_heading_rad)

        return frame


if __name__ == "__main__":
    odo = Odometry()

    try:
        while True:
            frame = odo.process_frame(1, 1, 1)

            cv2.imshow(odo.WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
