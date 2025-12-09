import cv2
import mediapipe as mp
import os

# ===== Config =====
# TODO: set this to the actual filename of your recording in the data/ folder
VIDEO_FILENAME = "kata_recording_20251209_125732.mp4"  # <-- change this
VIDEO_PATH = os.path.join("data", VIDEO_FILENAME)

# ==================

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

def main():
    if not os.path.exists(VIDEO_PATH):
        print(f"Error: Video file not found at {VIDEO_PATH}")
        return

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video: {VIDEO_PATH}")
        return

    # Pose estimator
    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        print("Press 'q' to quit.")
        while True:
            ret, frame = cap.read()
            if not ret:
                print("End of video or failed to read frame.")
                break

            # Convert BGR -> RGB for MediaPipe
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Improve performance (optional)
            image_rgb.flags.writeable = False
            results = pose.process(image_rgb)
            image_rgb.flags.writeable = True

            # Back to BGR for OpenCV display
            frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

            # Draw pose landmarks if detected
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame_bgr,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                )

            cv2.imshow("27 Movements - Pose Overlay", frame_bgr)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
