import math
import time
import cv2
from random import randint

from navigator.trackable_objects import Algae, Coral
from odometry.odometry import Odometry

def robot_pose_at_time(odo: Odometry, t):
    """Return (x,y,heading) for a looping figure-8 path."""
    cx = odo.FIELD_WIDTH_M / 2
    cy = odo.FIELD_HEIGHT_M / 2
    ax = odo.FIELD_WIDTH_M * 0.35
    ay = odo.FIELD_HEIGHT_M * 0.35
    omega = 0.5
    theta = omega * t
    x = cx + ax * math.sin(theta)
    y = cy + ay * math.sin(2 * theta) * 0.6
    dx = ax * omega * math.cos(theta)
    dy = ay * 2 * omega * math.cos(2 * theta) * 0.6
    heading = math.atan2(dy, dx)
    return x, y, heading

if __name__ == "__main__":
    odo = Odometry()

    for _ in range(3):
        algae = Algae()
        algae.update_relative_location(
            randint(500, 2000), randint(-60, 60))
        odo.game_pieces.add(Algae, algae)
    for _ in range(2):
        coral = Coral()
        coral.update_relative_location(
            randint(1000, 2500), randint(-90, 90))
        odo.game_pieces.add(Coral, coral)

    try:
        t0 = time.time()
        while True:
            Δt = time.time() - t0
            x, y, heading = robot_pose_at_time(odo, Δt)

            frame = odo.process_frame(x, y, heading)

            cv2.imshow(odo.WINDOW_NAME, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()