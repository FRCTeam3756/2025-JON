"""
Copyright (c) FRC Team 3756 RamFerno.
Open Source Software; you can modify and/or share it under the terms of
the license viewable in the root directory of this project.
"""

import math
import os
from typing import List, Tuple

import cv2
import numpy as np

from constants.field.apriltags import AprilTagConfig
from constants.robot.camera import CameraConfig
from constants.field.field import FieldConfig
from constants.robot.drivetrain import DriveTrainConfig
from src.navigator.trackable_objects import Algae, Cage, Coral, GamePieces, Robot

from logs.logging_setup import setup_logger


################################################

class Visualization:
    PIXELS_PER_METER = 70

    ROBOT_COLOR = (0, 125, 255)
    APRILTAG_COLOR = (50, 50, 50)
    DETECTION_COLOR_MAP = {
        Algae: (0, 255, 0),
        Cage: (255, 0, 0),
        Coral: (0, 255, 255),
        Robot: (255, 0, 255)
    }
    APRILTAG_CANVAS_SIZE_M = 0.35
    IMG_W = (
        int(FieldConfig.FIELD_WIDTH_M *
            PIXELS_PER_METER) + 2
    )
    IMG_H = (
        int(FieldConfig.FIELD_HEIGHT_M *
            PIXELS_PER_METER) + 2
    )
    
    def __init__(self) -> None:
        file_name = os.path.splitext(os.path.basename(__file__))[0]
        self.logger = setup_logger(file_name)
        self.logger.info("Odometry logger initialized.")

        self.game_pieces = GamePieces()
        
    def render_frame(self, robot_x_m: float, robot_y_m: float, robot_heading_rad: float, targets: List[Tuple[float, float]]) -> np.ndarray:
        canvas = np.zeros((self.IMG_H, self.IMG_W, 3), np.uint8)

        self.draw_field(canvas)
        self.draw_apriltags(canvas)
        self.draw_ramferno(canvas, robot_x_m, robot_y_m, robot_heading_rad)
        self.draw_vision_cone(canvas, robot_x_m, robot_y_m, robot_heading_rad)
        self.draw_detections(canvas, robot_x_m, robot_y_m, robot_heading_rad)
        for target in targets:
            self.draw_target_position(canvas, robot_x_m, robot_y_m, target[0], target[1])

        return canvas


    def camera_to_canvas(self, x_m: float, y_m: float) -> Tuple[int, int]:
        px = int(x_m * self.PIXELS_PER_METER)
        py = int(self.IMG_H - (y_m * self.PIXELS_PER_METER))
        return px, py

    def draw_field(self, canvas: np.ndarray) -> None:
        self.logger.debug("Drawing field.")
        top_left = (0, 0)
        bottom_right = (self.IMG_W,
                        self.IMG_H)
        canvas[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]] = (
            50,
            120,
            50,
        )
        canvas_x_px = int((FieldConfig.FIELD_WIDTH_M / 2) * self.PIXELS_PER_METER)
        cv2.line(
            canvas,
            (canvas_x_px, 0),
            (canvas_x_px, self.IMG_H),
            (200, 200, 200),
            1,
        )

    def draw_apriltags(self, canvas: np.ndarray) -> None:
        self.logger.debug("Drawing AprilTags on field.")
        half_px = int((self.APRILTAG_CANVAS_SIZE_M * self.PIXELS_PER_METER) / 2)
        for (tag_id, (x_m, y_m, rot_deg)) in (
            AprilTagConfig.APRILTAG_POSITIONS_M.items()
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
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            rad = math.radians(rot_deg)
            fx = int(px + math.cos(rad) * half_px * 1.3)
            fy = int(py - math.sin(rad) * half_px * 1.3)
            cv2.line(canvas, (px, py), (fx, fy), (255, 255, 255), 2)

    def draw_ramferno(self, canvas: np.ndarray, robot_x_m: float, robot_y_m: float, robot_heading_rad: float) -> None:
        self.logger.debug(
            f"Drawing robot at ({robot_x_m:.2f}, {robot_y_m:.2f}) with heading {math.degrees(robot_heading_rad):.1f}°")

        robot_field_x, robot_field_y = self.camera_to_canvas(
            robot_x_m, robot_y_m)

        robot_half_width_px = (
            DriveTrainConfig.ROBOT_WIDTH_M * self.PIXELS_PER_METER) / 2
        robot_half_length_px = (
            DriveTrainConfig.ROBOT_LENGTH_M * self.PIXELS_PER_METER) / 2
        corners = [
            (-robot_half_length_px, -robot_half_width_px),  # back left
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

    def draw_target_position(self, canvas: np.ndarray, robot_x_m: float, robot_y_m: float, target_x_m: float, target_y_m: float) -> None:
        robot_canvas_x, robot_canvas_y = self.camera_to_canvas(
            robot_x_m, robot_y_m)
        tx, ty = self.camera_to_canvas(target_x_m, target_y_m)
        cv2.circle(canvas, (tx, ty), 6, (0, 255, 0), -1)
        cv2.line(canvas, (robot_canvas_x, robot_canvas_y),
                 (tx, ty), (0, 200, 0), 1)
        dist = math.hypot(target_x_m - robot_x_m, target_y_m - robot_y_m)
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
                    self.logger.warning(
                        f"Skipping {cls.__name__}: Missing distance or angle data.")
                    continue

                obj_x_m, obj_y_m = self.object_world_coords(
                    obj.relative_distance_mm, obj.relative_angle_deg, robot_x_m, robot_y_m, robot_heading_rad
                )

                px, py = self.camera_to_canvas(obj_x_m, obj_y_m)

                self.logger.debug(
                    f"Drawing {cls.__name__} at ({px}, {py}) world=({obj_x_m:.2f},{obj_y_m:.2f})m")

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
        abs_angle_rad = robot_heading_rad - rel_angle_rad
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
            self.logger.info(
                f"Removed {removed} objects that stayed within the vision cone.")

