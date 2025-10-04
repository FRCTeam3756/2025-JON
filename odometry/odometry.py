import cv2
import numpy as np
import math
import time


class Odometry:
    FIELD_WIDTH_M = 54.0 * 0.3048  # 16.4592 m
    FIELD_HEIGHT_M = 27.0 * 0.3048  # 8.2296 m
    PIXELS_PER_METER = 70  # adjust for window size
    MARGIN_PX = 20
    ROBOT_RADIUS_M = 0.45
    ROBOT_COLOR = (0, 125, 255)
    ROBOT_ARROW_LENGTH_M = 1.0
    TAG_COLORS = [(255, 0, 0), (0, 255, 255)]
    INCH_TO_M = 0.0254

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
        self.duration_s = 20
        self.fps = 30
        self.frame_time = 1.0 / 30
        self.max_trail_points = 500
        self.trail = []
        self.target_pos = (self.FIELD_WIDTH_M * 0.85, self.FIELD_HEIGHT_M * 0.75)

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
        bottom_right = (self.IMG_W - self.MARGIN_PX, self.IMG_H - self.MARGIN_PX)
        canvas[top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]] = (
            50,
            120,
            50,
        )
        cv2.rectangle(canvas, top_left, bottom_right, (255, 255, 255), 2)
        cx_px = int(self.MARGIN_PX + (self.FIELD_WIDTH_M / 2) * self.PIXELS_PER_METER)
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

    def draw_robot(self, canvas, x_m, y_m, heading_rad):
        cx, cy = self.field_to_pixel(x_m, y_m)
        r_px = int(self.ROBOT_RADIUS_M * self.PIXELS_PER_METER)
        front = (r_px, 0)
        left = (-0.5 * r_px, 0.6 * r_px)
        right = (-0.5 * r_px, -0.6 * r_px)

        def rot(lx, ly):
            rx = lx * math.cos(heading_rad) - ly * math.sin(heading_rad)
            ry = lx * math.sin(heading_rad) + ly * math.cos(heading_rad)
            return int(cx + rx), int(cy - ry)

        p1, p2, p3 = rot(*front), rot(*left), rot(*right)
        pts = np.array([p1, p2, p3], np.int32)
        cv2.fillConvexPoly(canvas, pts, self.ROBOT_COLOR)
        cv2.polylines(canvas, [pts], True, (0, 0, 150), 2)

        arrow_len = int(self.ROBOT_ARROW_LENGTH_M * self.PIXELS_PER_METER)
        hx = int(cx + math.cos(heading_rad) * arrow_len)
        hy = int(cy - math.sin(heading_rad) * arrow_len)
        cv2.arrowedLine(canvas, (cx, cy), (hx, hy), (0, 0, 150), 2, tipLength=0.25)

        if len(self.trail) >= 2:
            pts_px = [self.field_to_pixel(x, y) for x, y in self.trail]
            for i in range(len(pts_px) - 1):
                thickness = max(1, int(3 * (i / len(pts_px))))
                cv2.line(canvas, pts_px[i], pts_px[i + 1], (0, 0, 180), thickness)

        if self.target_pos:
            tx_m, ty_m = self.target_pos
            tx, ty = self.field_to_pixel(tx_m, ty_m)
            cv2.circle(canvas, (tx, ty), 6, (0, 255, 0), -1)
            cv2.line(canvas, (cx, cy), (tx, ty), (0, 200, 0), 1)
            dist = math.hypot(tx_m - x_m, ty_m - y_m)
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

    def robot_pose_at_time(self, t):
        """Return (x,y,heading) for a looping figure-8 path."""
        cx = self.FIELD_WIDTH_M / 2
        cy = self.FIELD_HEIGHT_M / 2
        ax = self.FIELD_WIDTH_M * 0.35
        ay = self.FIELD_HEIGHT_M * 0.35
        omega = 0.5
        theta = omega * t
        x = cx + ax * math.sin(theta)
        y = cy + ay * math.sin(2 * theta) * 0.6
        dx = ax * omega * math.cos(theta)
        dy = ay * 2 * omega * math.cos(2 * theta) * 0.6
        heading = math.atan2(dy, dx)
        return x, y, heading

    def run(self):
        win = "FRC 2025 Odometry Demo"
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(win, self.IMG_W, self.IMG_H)
        t0 = time.time()
        total_frames = int(self.duration_s * self.fps)

        for f in range(total_frames):
            t = time.time() - t0
            x_m, y_m, heading = self.robot_pose_at_time(t)
            self.trail.append((x_m, y_m))
            if len(self.trail) > self.max_trail_points:
                self.trail.pop(0)

            canvas = np.zeros((self.IMG_H, self.IMG_W, 3), np.uint8)
            self.draw_field(canvas)
            self.draw_apriltags(canvas)
            self.draw_robot(canvas, x_m, y_m, heading)

            cv2.putText(
                canvas,
                f"t={t:.1f}s frame={f+1}/{total_frames}",
                (self.MARGIN_PX + 6, self.IMG_H - self.MARGIN_PX - 8),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

            cv2.imshow(win, canvas)
            if cv2.waitKey(1) & 0xFF == 27:
                break

            elapsed = time.time() - (t0 + f / self.fps)
            if (sleep := self.frame_time - elapsed) > 0:
                time.sleep(sleep)

        cv2.destroyAllWindows()

if __name__ == "__main__":
    odo = Odometry()
    odo.run()
