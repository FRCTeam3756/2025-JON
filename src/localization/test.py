from typing import List
import cv2
import numpy as np
from pupil_apriltags import Detector

from config import AprilTagConfig, FieldConfig


class PoseEstimator:
    def __init__(self, camera_matrix, dist_coeffs):
        self.detector = Detector(families='tag36h11')
        self.camera_matrix = camera_matrix
        self.dist_coeffs = dist_coeffs

    @staticmethod
    def invert_transform(R, t):
        R_inv = R.T
        t_inv = -R_inv @ t
        return t_inv

    def estimate_camera_pose(self, tag_R, tag_t, tag_field_pos):
        t_tc = self.invert_transform(tag_R, tag_t)

        R_ft = np.eye(3)
        t_ft = np.array(tag_field_pos).reshape(3, 1)

        t_fc = R_ft @ t_tc + t_ft
        return t_fc


class FieldVisualizer:
    def __init__(self):
        self.scale = 100
        self.img_width = int(FieldConfig.FIELD_WIDTH_M * self.scale)
        self.img_height = int(FieldConfig.FIELD_HEIGHT_M * self.scale)

    def draw_field(self, camera_positions=None):
        field_img = np.ones((self.img_height, self.img_width, 3), dtype=np.uint8) * 50

        cv2.rectangle(field_img, (0, 0),
                      (self.img_width - 1, self.img_height - 1),
                      (255, 255, 255), 2)

        for tag_id, pos in AprilTagConfig.APRILTAG_POSITIONS_M.items():
            if pos is None:
                continue
            x_m, y_m, _ = pos
            x_px = int(x_m * self.scale)
            y_px = int(self.img_height - y_m * self.scale)
            cv2.circle(field_img, (x_px, y_px), 8, (0, 255, 255), -1)
            cv2.putText(field_img, f"{tag_id}", (x_px + 10, y_px - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        if camera_positions:
            for cam in camera_positions:
                x, y, _ = cam
                x_px = int(x * self.scale)
                y_px = int(self.img_height - y * self.scale)
                cv2.circle(field_img, (x_px, y_px), 15, (0, 0, 255), -1)

        return field_img


class VideoPoseEstimator:
    def __init__(self, video_path, camera_matrix, dist_coeffs):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        self.pose_estimator = PoseEstimator(camera_matrix, dist_coeffs)
        self.visualizer = FieldVisualizer()

    def process(self):
        camera_positions = []
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break

            gray = np.astype(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), np.uint8)
            tags = self.pose_estimator.detector.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(
                    self.pose_estimator.camera_matrix[0, 0],
                    self.pose_estimator.camera_matrix[1, 1],
                    self.pose_estimator.camera_matrix[0, 2],
                    self.pose_estimator.camera_matrix[1, 2]
                ),
                tag_size=AprilTagConfig.APRILTAG_SIZE_MM / 1000
            )
            if not isinstance(tags, List):
                tags = [tags]

            camera_positions = []

            for tag in tags:
                tag_id = tag.tag_id
                if tag_id not in AprilTagConfig.APRILTAG_POSITIONS_M:
                    print(f"[Warning]: Unknown AprilTag ID {tag_id}, skipping.")
                    continue
                
                field_pose = AprilTagConfig.APRILTAG_POSITIONS_M[tag_id]

                if field_pose is None:
                    continue

                t_fc = self.pose_estimator.estimate_camera_pose(tag.pose_R, tag.pose_t, field_pose)
                camera_positions.append(t_fc.ravel())

                assert tag.corners is not None, f"Tag {tag.tag_id} has no corner data."

                for c in tag.corners.astype(int):
                    cv2.circle(frame, tuple(c), 4, (0, 255, 0), -1)
                cv2.putText(frame, f"ID {tag_id}", tuple(tag.corners[0].astype(int)), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            if camera_positions:
                avg_pos = np.mean(camera_positions, axis=0)
                x, y, z = avg_pos
                cv2.putText(frame, f"Camera: x={x:.2f}m, y={y:.2f}m, z={z:.2f}m",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                print(f"Camera position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

            field_img = self.visualizer.draw_field(camera_positions)

            cv2.imshow("Pose Estimation", frame)
            cv2.imshow("Field Map", field_img)
            if cv2.waitKey(1) == 27:  # ESC key
                break

        self.cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    CAMERA_MATRIX = np.array([[600, 0, 320],
                              [0, 600, 240],
                              [0,   0,   1]], dtype=float)
    DIST_COEFFS = np.zeros((4, 1))

    video_path = "test/input/video.mp4"

    estimator = VideoPoseEstimator(video_path, CAMERA_MATRIX, DIST_COEFFS)
    estimator.process()