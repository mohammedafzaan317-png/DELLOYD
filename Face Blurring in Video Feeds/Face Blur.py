import cv2
import datetime
import sys

def get_face_detector():
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise IOError(f"Failed to load face cascade from: {cascade_path}")
    return detector

def blur_faces(frame, faces, blur_kernel=(51,51)):
    for (x, y, w, h) in faces:
        margin = int(0.1 * min(w, h))  # Slight margin around face
        x1 = max(0, x - margin)
        y1 = max(0, y - margin)
        x2 = min(frame.shape[1], x + w + margin)
        y2 = min(frame.shape[0], y + h + margin)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        blurred = cv2.GaussianBlur(roi, blur_kernel, 0)
        frame[y1:y2, x1:x2] = blurred
    return frame

def main(video_source=0):
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print(f"Error: cannot open video source {video_source}")
        return

    face_detector = get_face_detector()

    recording = False
    out = None

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0 or fps is None:
        fps = 20.0

    print(f"Video source opened: width={width}, height={height}, fps={fps}")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame, exiting.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )

            processed = blur_faces(frame, faces)

            if recording and out is not None:
                out.write(processed)

            cv2.imshow("Face Blurred Live", processed)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[INFO] Quit signal received.")
                break
            elif key == ord('s'):
                if not recording:
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"clip_{timestamp}.avi"
                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                    if not out.isOpened():
                        print("Error: unable to open output video writer.")
                        out = None
                        recording = False
                    else:
                        recording = True
                        print(f"[INFO] Recording started: {filename}")
                else:
                    recording = False
                    if out:
                        out.release()
                        out = None
                    print("[INFO] Recording stopped.")

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user, exiting...")

    finally:
        print("[INFO] Cleaning up...")
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main(video_source=0)
