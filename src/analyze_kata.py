import cv2
import mediapipe as mp
import os

from src.moves.move_analyzer import MoveAnalyzer

# ===== Config =====
VIDEO_FILENAME = "kata_recording_20251209_125732.mp4"
VIDEO_PATH = os.path.join("data", VIDEO_FILENAME)

PLAYBACK_DELAY_MS = 33  # ~30 FPS; increase to slow down, decrease to speed up
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

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps is None or fps <= 0:
        fps = 30.0

    move_analyzer = MoveAnalyzer(fps=fps)

    paused = False
    step = False
    frame_count = 0

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        enable_segmentation=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as pose:

        print("Controls:")
        print("  q - quit")
        print("  p or space - pause / resume")
        print("  n - next frame (step one frame when paused)")
        print("  d - print current debug metrics to terminal")
        print("Press 'p' or spacebar to pause at any time.")

        last_frame_bgr = None
        last_landmarks = None

        while True:
            # Only read a new frame when not paused OR when stepping
            if not paused or step:
                ret, frame = cap.read()
                if not ret:
                    print("End of video or failed to read frame.")
                    break

                frame_count += 1
                height, width, _ = frame.shape

                # Start a new debug record for this frame
                move_analyzer.reset_debug_data()

                # BGR -> RGB
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_rgb.flags.writeable = False
                results = pose.process(image_rgb)
                image_rgb.flags.writeable = True
                frame_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

                landmarks = results.pose_landmarks.landmark if results.pose_landmarks else None

                # Draw and analyze when we have landmarks
                if landmarks:
                    mp_drawing.draw_landmarks(
                        frame_bgr,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style(),
                    )

                    # ---- high-level movement analysis ----
                    
                    # 1) stance (baseline for everything else)
                    move_analyzer.analyze_horse_stance(
                        frame_bgr,
                        landmarks,
                        frame_count,
                        width,
                        height,
                    )

                    # 2) yoi (ready position)
                    move_analyzer.analyze_yoi(
                        frame_bgr,
                        landmarks,
                        width,
                        height,
                    )

                    move_analyzer.analyze_upper_block(
                        frame_bgr,
                        landmarks,
                        width,
                        height,
                    )
                    
                    # move_analyzer.analyze_center_punch(
                    #     frame_bgr,
                    #     landmarks,
                    #     width,
                    #     height,
                    # )

                    # move_analyzer.analyze_over_shoulder_punch(
                    #     frame_bgr,
                    #     landmarks,
                    #     width,
                    #     height,
                    # )
                    

                # store last processed frame + landmarks for paused replay
                last_frame_bgr = frame_bgr
                last_landmarks = landmarks

                # consume the one-step flag
                step = False

            else:
                # When paused and not stepping, just reuse the last frame
                if last_frame_bgr is None:
                    continue
                frame_bgr = last_frame_bgr
                height, width = frame_bgr.shape[:2]

            # Show frame (either new or last)
            cv2.imshow("27 Movements - Modular Analyzer (with playback controls)", frame_bgr)

            # waitKey depends on paused/running state
            delay = 0 if paused else PLAYBACK_DELAY_MS
            key = cv2.waitKey(delay) & 0xFF

            if key == ord("q"):
                break
            elif key in (ord("p"), 32):  # 'p' or spacebar
                paused = not paused
                print(f"{'Paused' if paused else 'Resumed'} at frame {frame_count}")
            elif key == ord("n"):
                paused = True
                step = True
            elif key == ord("d"):
                # Print current debug metrics to terminal
                snapshot = move_analyzer.get_debug_snapshot()
                print(f"\n[Debug] Frame {frame_count}")
                if not snapshot:
                    print("  (no debug data recorded)")
                else:
                    for k, v in snapshot.items():
                        print(f"  {k}: {v}")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
