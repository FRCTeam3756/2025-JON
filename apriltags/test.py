import cv2
import robotpy_apriltag as apriltag
import ntcore

FOCAL_LENGTH = 800
TAG_SIZE_CM = 9 * 2.54
LINE_LENGTH = 10

def estimate_distance(tag_size_pixels):
    """Estimate distance to the tag based on its size in the image."""
    distance = (FOCAL_LENGTH * TAG_SIZE_CM) / tag_size_pixels
    return distance
9
def send_drive_instructions(forward, strafe, rotation):
    """Send drive instructions to NetworkTables."""
    # vision_table.putNumber("drive_forward", forward)
    # vision_table.putNumber("drive_strafe", strafe)
    # vision_table.putNumber("drive_rotation", rotation)
    print(f"Sent Drive Instructions - Forward: {forward}, Strafe: {strafe}, Rotation: {rotation}")

def main():
    """Main loop for processing frames and sending drive instructions."""    
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)

    detector = apriltag.AprilTagDetector()
    detector.addFamily("tag36h11", 3)

    tagsTable = ntcore.NetworkTableInstance.getDefault().getTable("apriltags")
    pubTags = tagsTable.getIntegerArrayTopic("tags").publish()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # frame = cv2.flip(frame, 1)
        greyFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        detections = detector.detect(greyFrame)

        tags = []

        for detection in detections:
            tags.append(detection.getId())

            for i in range(4):
                j = (i + 1) % 4
                point1 = (int(detection.getCorner(i).x), int(detection.getCorner(i).y))
                point2 = (int(detection.getCorner(j).x), int(detection.getCorner(j).y))
                cv2.line(frame, point1, point2, (0, 255, 0), 2)

            center_x = int(detection.getCenter().x)
            center_y = int(detection.getCenter().y)
            
            cv2.line(frame, (center_x - LINE_LENGTH, center_y), (center_x + LINE_LENGTH, center_y), (0, 0, 255), 2)
            cv2.line(frame, (center_x, center_y - LINE_LENGTH), (center_x, center_y + LINE_LENGTH), (0, 0, 255), 2)
            cv2.putText(frame, str(detection.getId()), (center_x + LINE_LENGTH, center_y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

        pubTags.set(tags)

        cv2.imshow("AprilTags", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyALINE_LENGTHWindows()

if __name__ == "__main__":
    main()
