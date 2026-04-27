import cv2
import mediapipe as mp
import math


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


def calculate_angle(a, b, c):
    angle = math.degrees(
        math.atan2(c[1] - b[1], c[0] - b[0]) -
        math.atan2(a[1] - b[1], a[0] - b[0])
    )

    angle = abs(angle)

    if angle > 180:
        angle = 360 - angle

    return angle


def get_landmark(landmarks, landmark_name, width, height):
    landmark = landmarks[landmark_name.value]
    return int(landmark.x * width), int(landmark.y * height)


def main():
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while True:
            success, frame = cap.read()

            if not success:
                print("Camera not detected")
                break

            frame = cv2.flip(frame, 1)
            height, width, _ = frame.shape

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb_frame)

            status = "No person detected"
            color = (0, 0, 255)

            if result.pose_landmarks:
                landmarks = result.pose_landmarks.landmark

                left_shoulder = get_landmark(
                    landmarks,
                    mp_pose.PoseLandmark.LEFT_SHOULDER,
                    width,
                    height
                )
                right_shoulder = get_landmark(
                    landmarks,
                    mp_pose.PoseLandmark.RIGHT_SHOULDER,
                    width,
                    height
                )
                left_ear = get_landmark(
                    landmarks,
                    mp_pose.PoseLandmark.LEFT_EAR,
                    width,
                    height
                )
                left_hip = get_landmark(
                    landmarks,
                    mp_pose.PoseLandmark.LEFT_HIP,
                    width,
                    height
                )

                shoulder_difference = abs(left_shoulder[1] - right_shoulder[1])
                neck_angle = calculate_angle(left_ear, left_shoulder, left_hip)

                if shoulder_difference < 25 and 150 <= neck_angle <= 180:
                    status = "Good Posture"
                    color = (0, 255, 0)
                else:
                    status = "Bad Posture - Sit Straight"
                    color = (0, 0, 255)

                cv2.putText(
                    frame,
                    f"Neck Angle: {int(neck_angle)}",
                    (30, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                cv2.putText(
                    frame,
                    f"Shoulder Difference: {shoulder_difference}px",
                    (30, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    color,
                    2
                )

                mp_drawing.draw_landmarks(
                    frame,
                    result.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

            cv2.putText(
                frame,
                status,
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2
            )

            cv2.imshow("AI-Based Posture Detection", frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
