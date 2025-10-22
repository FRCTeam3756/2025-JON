from typing import List
import cv2
import numpy as np
from pupil_apriltags import Detector

from config import AprilTagConfig, FieldConfig
from visualization.visualization import Visualization


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


class VideoPoseEstimator:
    def __init__(self, video_path, camera_matrix, dist_coeffs):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise IOError(f"Could not open video: {video_path}")
        self.pose_estimator = PoseEstimator(camera_matrix, dist_coeffs)
        self.visualization = Visualization()

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

            for position in camera_positions:
                x, y, z = position
                cv2.putText(frame, f"Camera: x={x:.2f}m, y={y:.2f}m, z={z:.2f}m",
                            (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                print(f"Camera position: x={x:.2f}, y={y:.2f}, z={z:.2f}")

            field_img = self.visualization.draw_field()

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