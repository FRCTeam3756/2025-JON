import math
import tkinter as tk
from config import CameraConfig, FieldConfig, AprilTagConfig, RamFernoRobotConfig
from src.localization.localization import Localization


class FieldVisualizer:
    SCALE = 70

    def __init__(self, root):
        self.root = root
        self.root.title("AprilTag Localization Visualizer")
        self.config = AprilTagConfig()
        self.localization = Localization()

        self.canvas = tk.Canvas(root, width=(FieldConfig.FIELD_WIDTH_M * self.SCALE), height=(FieldConfig.FIELD_HEIGHT_M * self.SCALE), bg="green")
        self.canvas.pack()

        self.selected_tag = None
        self.robot_marker = None
        self.direction_indicator = None
        self.vision_line = None
        self.robot_to_object_line = None

        self.distance_slider = tk.Scale(root, from_=0, to=10, resolution=0.01,
                                        orient="horizontal", label="Relative Distance (m)", command=self.update_robot)
        self.distance_slider.pack(fill="x")

        self.angle_slider = tk.Scale(root, from_=(-CameraConfig.HORIZONTAL_FOV_DEG / 2), to=(CameraConfig.HORIZONTAL_FOV_DEG / 2), orient="horizontal",
                                     label="Relative Angle (°)", command=self.update_robot)
        self.angle_slider.pack(fill="x")

        self.tag_drawings = {}
        self.draw_apriltags()

        self.canvas.bind("<Button-1>", self.on_click)

        self.position_label = tk.Label(root, text="Robot Position: (x=?, y=?)", anchor="w", font=("Arial", 10))
        self.position_label.pack(fill="x")

        self.robot_half_width = (RamFernoRobotConfig.ROBOT_WIDTH_M * self.SCALE) / 2
        self.robot_half_height = (RamFernoRobotConfig.ROBOT_LENGTH_M * self.SCALE) / 2

    def draw_apriltags(self):
        for tag_id, (x_m, y_m, _) in self.config.APRILTAG_POSITIONS_M.items():
            x = x_m * self.SCALE
            y = FieldConfig.FIELD_HEIGHT_M * self.SCALE - y_m * self.SCALE
            size = 10
            tag = self.canvas.create_rectangle(x - size, y - size, x + size, y + size, fill="gray")
            self.canvas.create_text(x, y - 15, text=str(tag_id), font=("Arial", 8))
            self.tag_drawings[tag_id] = tag

    def on_click(self, event):
        for tag_id, rect in self.tag_drawings.items():
            x1, y1, x2, y2 = self.canvas.coords(rect)
            if x1 <= event.x <= x2 and y1 <= event.y <= y2:
                self.selected_tag = tag_id
                self.canvas.itemconfig(rect, fill="red")
                for other_id, other_rect in self.tag_drawings.items():
                    if other_id != tag_id:
                        self.canvas.itemconfig(other_rect, fill="gray")
                self.update_robot()
                break

    def draw_robot(self, corners, front_center_x, front_center_y, vision_end_x, vision_end_y, tag_x_px, tag_y_px):
        self.robot_marker = self.canvas.create_polygon(
            corners, fill="orange", outline="black"
        )
        self.direction_indicator = self.canvas.create_line(corners[1][0], corners[1][1], corners[2][0], corners[2][1], fill="red", width=3)
        self.vision_line = self.canvas.create_line(front_center_x, front_center_y, vision_end_x, vision_end_y, fill="black", width=3)
        self.robot_to_object_line = self.canvas.create_line(front_center_x, front_center_y, tag_x_px, tag_y_px, fill="black", width=3)

    def clean_up_drawings(self):
        if self.robot_marker:
            self.canvas.delete(self.robot_marker)
        if self.direction_indicator:
            self.canvas.delete(self.direction_indicator)
        if self.vision_line:
            self.canvas.delete(self.vision_line)
        if self.robot_to_object_line:
            self.canvas.delete(self.robot_to_object_line)

    def update_robot(self, _=None):
        if self.selected_tag is None:
            return

        rel_dist = self.distance_slider.get()
        rel_angle_deg = self.angle_slider.get()
        rel_angle_rad = math.radians(rel_angle_deg)

        robot_x_m, robot_y_m, robot_heading_rad = self.localization.get_world_position(
            self.selected_tag, rel_dist, rel_angle_deg
        )

        robot_absolute_heading_rad = robot_heading_rad + rel_angle_rad

        self.clean_up_drawings()

        robot_x_px = robot_x_m * self.SCALE
        robot_y_px = FieldConfig.FIELD_HEIGHT_M * self.SCALE - robot_y_m * self.SCALE

        corners = []
        for dx, dy in [(-self.robot_half_width, -self.robot_half_height), (self.robot_half_width, -self.robot_half_height),
                    (self.robot_half_width, self.robot_half_height), (-self.robot_half_width, self.robot_half_height)]:
            rotated_x = robot_x_px + (dx * math.cos(robot_absolute_heading_rad) - dy * math.sin(robot_absolute_heading_rad))
            rotated_y = robot_y_px + (dx * math.sin(robot_absolute_heading_rad) + dy * math.cos(robot_absolute_heading_rad))
            corners.append((rotated_x, rotated_y))

        front_center_x = (corners[1][0] + corners[2][0]) / 2
        front_center_y = (corners[1][1] + corners[2][1]) / 2

        vision_end_x = front_center_x + math.cos(robot_absolute_heading_rad) * rel_dist * self.SCALE
        vision_end_y = front_center_y + -math.sin(robot_absolute_heading_rad) * rel_dist * self.SCALE

        tag_x_m, tag_y_m, tag_angle_deg = self.config.APRILTAG_POSITIONS_M[self.selected_tag]
        tag_x_px = tag_x_m * self.SCALE
        tag_y_px = FieldConfig.FIELD_HEIGHT_M * self.SCALE - tag_y_m * self.SCALE

        self.draw_robot(corners, front_center_x, front_center_y, vision_end_x, vision_end_y, tag_x_px, tag_y_px)

        self.position_label.config(
            text=f"Robot Position: X = {robot_x_m:.2f} m, Y = {robot_y_m:.2f} m, Heading = {math.degrees(robot_absolute_heading_rad):.1f}°"
        )

if __name__ == "__main__":
    root = tk.Tk()
    app = FieldVisualizer(root)
    root.mainloop()
