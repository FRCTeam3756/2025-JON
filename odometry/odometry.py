import cv2
import math
import numpy as np

from navigator.trackable_objects import Algae, Coral, Cage, GamePieces, Robot


class Odometry:
    FIELD_WIDTH_M = 16.4592
    FIELD_HEIGHT_M = 8.2296
    PIXELS_PER_METER = 70
    MARGIN_PX = 20
    ROBOT_RADIUS_M = 0.45
    ROBOT_COLOR = (0, 125, 255)
    ROBOT_ARROW_LENGTH_M = 1.0
    TAG_COLORS = [(255, 0, 0), (0, 255, 255)]
    INCH_TO_M = 0.0254
    VISION_FOV_DEG = 60
    VISION_RANGE_M = 4.0
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

    def __init__(self):
        self.fps = 30
        self.target_x = self.FIELD_WIDTH_M * 0.85
        self.target_y = self.FIELD_HEIGHT_M * 0.75
        self.game_pieces = GamePieces()

        self.APRILTAG_POSITIONS = {
            k: (x * self.INCH_TO_M, y * self.INCH_TO_M, r)
            for k, (x, y, r) in self.APRILTAG_POSITIONS.items()
        }

        self.IMG_W = (
            int(self.FIELD_WIDTH_M * self.PIXELS_PER_METER) + 2 * self.MARGIN_PX
        )
        self.IMG_H = (
            int(self.FIELD_HEIGHT_M * self.PIXELS_PER_METER) + 2 * self.MARGIN_PX
        )

    def field_to_pixel(self, x_m, y_m):
        px = int(self.MARGIN_PX + x_m * self.PIXELS_PER_METER)
        py = int(self.IMG_H - (self.MARGIN_PX + y_m * self.PIXELS_PER_METER))
        return px, py

    def draw_field(self, canvas):
        top_left = (self.MARGIN_PX, self.MARGIN_PX)
        bottom_right = (self.IMG_W - self.MARGIN_PX,
                        self.IMG_H - self.MARGIN_PX)
        canvas[top_left[1]: bottom_right[1], top_left[0]: bottom_right[0]] = (
            50,
            120,
            50,
        )
        cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), 2)
        cx_px = int(self.MARGIN_PX + (self.FIELD_WIDTH_M / 2)
                    * self.PIXELS_PER_METER)
        cv2.line(
            canvas,
            (cx_px, self.MARGIN_PX),
            (cx_px, self.IMG_H - self.MARGIN_PX),
            (200, 200, 200),
            1,
        )
        cv2.putText(
            canvas,
            f"Field: {self.FIELD_WIDTH_M:.2f}m x {self.FIELD_HEIGHT_M:.2f}m",
            (self.MARGIN_PX + 6, self.MARGIN_PX + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (230, 230, 230),
            1,
            cv2.LINE_AA,
        )

    def draw_apriltags(self, canvas):
        tag_size_m = 0.35
        half_px = int((tag_size_m * self.PIXELS_PER_METER) / 2)
        for i, (tag_id, (x_m, y_m, rot_deg)) in enumerate(
            self.APRILTAG_POSITIONS.items()
        ):
            px, py = self.field_to_pixel(x_m, y_m)
            color = self.TAG_COLORS[i % len(self.TAG_COLORS)]
            cv2.rectangle(
                canvas,
                (px - half_px, py - half_px),
                (px + half_px, py + half_px),
                color,
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

    def draw_ramferno(self, canvas, x_m, y_m, heading_rad):
        cx, cy = self.field_to_pixel(x_m, y_m)
        r_px = int(self.ROBOT_RADIUS_M * self.PIXELS_PER_METER)
        half_side = r_px
        corners = [
            (-half_side, -half_side),
            (half_side, -half_side),
            (half_side, half_side),
            (-half_side, half_side),
        ]

        def rot(lx, ly):
            rx = lx * math.cos(heading_rad) - ly * math.sin(heading_rad)
            ry = lx * math.sin(heading_rad) + ly * math.cos(heading_rad)
            return int(cx + rx), int(cy - ry)

        pts = np.array([rot(x, y) for x, y in corners], np.int32)
        cv2.fillConvexPoly(canvas, pts, self.ROBOT_COLOR)
        cv2.polylines(canvas, [pts], True, (0, 0, 150), 2)

        front_local = [(half_side, -half_side), (half_side, half_side)]
        front_pts = np.array([rot(x, y) for x, y in front_local], np.int32)
        cv2.line(canvas, tuple(front_pts[0]),
                 tuple(front_pts[1]), (0, 0, 255), 3)

        tx, ty = self.field_to_pixel(self.target_x, self.target_y)
        cv2.circle(canvas, (tx, ty), 6, (0, 255, 0), -1)
        cv2.line(canvas, (cx, cy), (tx, ty), (0, 200, 0), 1)
        dist = math.hypot(self.target_x - x_m, self.target_y - y_m)
        midx, midy = (cx + tx) // 2, (cy + ty) // 2
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

    def draw_objects(self, canvas, robot_x, robot_y, robot_heading):
        """Draw all additional tracked objects."""
        color_map = {
            Algae: (0, 255, 0),
            Cage: (255, 0, 0),
            Coral: (0, 255, 255),
            Robot: (255, 0, 255)
        }

        for cls, objs in list(self.game_pieces._data.items()):
            for obj in list(objs):
                if obj.distance_in_mm is None or obj.angle_in_degrees is None:
                    continue

                dist_m = obj.distance_in_mm * 0.001
                rel_angle_rad = math.radians(obj.angle_in_degrees)
                abs_angle_rad = robot_heading + rel_angle_rad
                obj_x_m = robot_x + dist_m * math.cos(abs_angle_rad)
                obj_y_m = robot_y + dist_m * math.sin(abs_angle_rad)
                
                px, py = self.field_to_pixel(obj_x_m, obj_y_m)
                cv2.circle(canvas, (px, py), 6, color_map.get(cls, (200, 200, 200)), -1)
                cv2.putText(
                    canvas,
                    cls.__name__,
                    (px + 8, py - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    color_map.get(cls, (200, 200, 200)),
                    1,
                    cv2.LINE_AA,
                )

    def update_objects(self, fov_rad, range_m, robot_x_m, robot_y_m, robot_heading_rad) -> None:
        for cls, objs in list(self.game_pieces._data.items()):
            for obj in list(objs):
                if obj.distance_in_mm is None or obj.angle_in_degrees is None:
                    continue
                
                dist_m = obj.distance_in_mm * 0.001
                rel_angle_rad = math.radians(obj.angle_in_degrees)
                abs_angle_rad = robot_heading_rad + rel_angle_rad
                obj_x_m = robot_x_m + dist_m * math.cos(abs_angle_rad)
                obj_y_m = robot_y_m + dist_m * math.sin(abs_angle_rad)

                visible = self.object_visible(robot_x_m, robot_y_m, robot_heading_rad, obj_x_m, obj_y_m, fov_rad, range_m)

                if not visible:
                    self.game_pieces._data[cls].remove(obj)

    def draw_vision_cone(self, canvas, x_m, y_m, heading_rad):
        """Draws a semi-transparent grey cone representing robot vision."""
        fov = math.radians(60)
        range_m = 4.0

        left_angle = heading_rad - fov / 2
        right_angle = heading_rad + fov / 2

        cx, cy = self.field_to_pixel(x_m, y_m)
        end_left = self.field_to_pixel(
            x_m + range_m * math.cos(left_angle),
            y_m + range_m * math.sin(left_angle)
        )
        end_right = self.field_to_pixel(
            x_m + range_m * math.cos(right_angle),
            y_m + range_m * math.sin(right_angle)
        )

        cone_pts = np.array([ [cx, cy], end_left, end_right ], np.int32)

        overlay = canvas.copy()
        cv2.fillConvexPoly(overlay, cone_pts, (100, 100, 100))
        cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

        return fov, range_m

    def object_visible(self, robot_x_m, robot_y_m, robot_heading_rad, obj_x_m, obj_y_m, fov_rad, range_m):
        """Checks if object is within robot's vision cone."""
        dx = obj_x_m - robot_x_m
        dy = obj_y_m - robot_y_m
        dist = math.hypot(dx, dy)
        if dist > range_m:
            return False
        
        angle_to_obj = math.atan2(dy, dx)
        angle_diff = (angle_to_obj - robot_heading_rad + math.pi) % (2 * math.pi) - math.pi
        
        return abs(angle_diff) <= fov_rad / 2
    
    def initialize_window(self) -> None:
        cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.WINDOW_NAME, self.IMG_W, self.IMG_H)

    def process_frame(self, robot_x: float, robot_y: float, robot_heading: float) -> np.ndarray:
        canvas = np.zeros((self.IMG_H, self.IMG_W, 3), np.uint8)
        self.draw_field(canvas)
        self.draw_apriltags(canvas)
        self.draw_ramferno(canvas, robot_x, robot_y, robot_heading)
        
        fov_rad, range_m = self.draw_vision_cone(canvas, robot_x, robot_y, robot_heading)
        self.update_objects(fov_rad, range_m, robot_x, robot_y, robot_heading)
        self.draw_objects(canvas, robot_x, robot_y, robot_heading)

        return canvas

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