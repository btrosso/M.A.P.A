import cv2
import time

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    recording = False
    out = None
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    print("Controls:")
    print("  r - start recording")
    print("  s - stop recording")
    print("  q - quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame from webcam.")
            break

        # If recording, write frames to output file
        if recording and out is not None:
            out.write(frame)

            # Show indicator on screen
            cv2.putText(
                frame,
                "REC",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 255),
                3,
                cv2.LINE_AA
            )

        # Show webcam feed
        cv2.imshow("Webcam - Recorder", frame)

        key = cv2.waitKey(1) & 0xFF

        # Start recording
        if key == ord("r") and not recording:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"kata_recording_{timestamp}.mp4"
            filepath = f"data/{filename}"

            # Video writer — match webcam resolution + FPS
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps == 0:
                fps = 30  # fallback if camera doesn't report FPS

            out = cv2.VideoWriter(filepath, fourcc, fps, (frame_width, frame_height))

            recording = True
            print(f"[+] Recording started → {filepath}")

        # Stop recording
        elif key == ord("s") and recording:
            recording = False
            if out:
                out.release()
                out = None
            print("[+] Recording stopped.")

        # Quit program
        elif key == ord("q"):
            break

    # Cleanup
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
