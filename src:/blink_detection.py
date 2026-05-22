"""
Blink detection module for the eye tracking project.

This script uses MediaPipe Face Mesh landmarks to calculate an Eye Aspect
Ratio-like eye openness value. It calibrates a blink threshold for the user,
filters short noise with consecutive-frame control, and displays blink status.
"""

import time

import cv2
import mediapipe as mp
import numpy as np


CAMERA_INDEX = 0
WINDOW_NAME = "Blink Detection - Press Q to Quit"

CALIBRATION_SECONDS = 3
MIN_CALIBRATION_SAMPLES = 15
THRESHOLD_SCALE = 0.75
CONSECUTIVE_BLINK_FRAMES = 3
BLINK_COOLDOWN_SECONDS = 0.6

SUCCESS_COLOR = (0, 255, 0)
WARNING_COLOR = (0, 255, 255)
ERROR_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)

# EAR-style landmark order:
# horizontal points: p1, p4
# vertical pairs: p2-p6 and p3-p5
LEFT_EYE_EAR_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_POINTS = [362, 385, 387, 263, 373, 380]

LEFT_EYE_CONTOUR = [
    33,
    246,
    161,
    160,
    159,
    158,
    157,
    173,
    133,
    155,
    154,
    153,
    145,
    144,
    163,
    33,
]
RIGHT_EYE_CONTOUR = [
    362,
    398,
    384,
    385,
    386,
    387,
    388,
    466,
    263,
    249,
    390,
    373,
    374,
    380,
    381,
    382,
    362,
]

def open_camera(camera_index=CAMERA_INDEX):
    """Open the selected webcam and return the capture object."""
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise RuntimeError(
            "Camera could not be opened. Check webcam connection or camera index."
        )

    return camera


def create_face_mesh():
    """Create and return a MediaPipe Face Mesh detector."""
    face_mesh_module = mp.solutions.face_mesh

    return face_mesh_module.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

def detect_face_landmarks(frame, face_mesh):
    """Detect face landmarks in a BGR frame."""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_frame.flags.writeable = False

    results = face_mesh.process(rgb_frame)

    return results.multi_face_landmarks


def landmark_to_pixel(landmark, frame_width, frame_height):
    """Convert a normalized MediaPipe landmark to pixel coordinates."""
    x = int(landmark.x * frame_width)
    y = int(landmark.y * frame_height)
    return np.array([x, y])


def get_landmark_pixels(face_landmarks, indexes, frame_width, frame_height):
    """Convert selected landmark indexes to pixel coordinates."""
    points = []

    for index in indexes:
        landmark = face_landmarks.landmark[index]
        points.append(landmark_to_pixel(landmark, frame_width, frame_height))

    return points

def calculate_eye_aspect_ratio(points):
    """Calculate an EAR-like eye openness ratio from six eye points."""
    p1, p2, p3, p4, p5, p6 = points

    vertical_distance_1 = np.linalg.norm(p2 - p6)
    vertical_distance_2 = np.linalg.norm(p3 - p5)
    horizontal_distance = np.linalg.norm(p1 - p4)

    if horizontal_distance == 0:
        return 0

    ear = (vertical_distance_1 + vertical_distance_2) / (2 * horizontal_distance)
    return ear

def calculate_average_ear(frame, face_landmarks):
    """Calculate the average EAR value of both eyes."""
    frame_height, frame_width, _ = frame.shape

    left_eye_points = get_landmark_pixels(
        face_landmarks,
        LEFT_EYE_EAR_POINTS,
        frame_width,
        frame_height,
    )
    right_eye_points = get_landmark_pixels(
        face_landmarks,
        RIGHT_EYE_EAR_POINTS,
        frame_width,
        frame_height,
    )

    left_ear = calculate_eye_aspect_ratio(left_eye_points)
    right_ear = calculate_eye_aspect_ratio(right_eye_points)

    return (left_ear + right_ear) / 2

def draw_eye_contours(frame, face_landmarks):
    """Draw visible contours around both eyes."""
    frame_height, frame_width, _ = frame.shape

    for contour, color in [
        (LEFT_EYE_CONTOUR, (0, 255, 255)),
        (RIGHT_EYE_CONTOUR, (255, 0, 255)),
    ]:
        points = get_landmark_pixels(face_landmarks, contour, frame_width, frame_height)

        for point in points:
            cv2.circle(frame, tuple(point), 3, color, -1)

        for start_point, end_point in zip(points, points[1:]):
            cv2.line(frame, tuple(start_point), tuple(end_point), color, 2)

    return frame

def put_text(frame, text, position, color=TEXT_COLOR, scale=0.8, thickness=2):
    """Draw text on the frame."""
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )
    return frame


def calibrate_threshold(calibration_values):
    """Create a user-specific blink threshold from open-eye EAR samples."""
    open_eye_average = float(np.mean(calibration_values))
    threshold = open_eye_average * THRESHOLD_SCALE
    return threshold, open_eye_average

def update_blink_state(
    ear,
    threshold,
    consecutive_closed_frames,
    last_blink_time,
    blink_count,
):
    """Detect blink events using thresholding, frame filtering, and cooldown."""
    current_time = time.time()
    blink_detected = False

    if ear < threshold:
        consecutive_closed_frames += 1
    else:
        consecutive_closed_frames = 0

    cooldown_finished = current_time - last_blink_time > BLINK_COOLDOWN_SECONDS
    enough_closed_frames = consecutive_closed_frames >= CONSECUTIVE_BLINK_FRAMES

    if enough_closed_frames and cooldown_finished:
        blink_detected = True
        blink_count += 1
        last_blink_time = current_time
        print(f"Blink count: {blink_count}")

    return blink_detected, consecutive_closed_frames, last_blink_time, blink_count

def draw_runtime_info(
    frame,
    face_detected,
    calibrated,
    blink_detected,
    ear,
    threshold,
    blink_count,
    calibration_start_time,
):
    """Draw calibration, face, EAR, and blink information on the frame."""
    if not face_detected:
        return put_text(
            frame,
            "No face detected",
            (20, 60),
            ERROR_COLOR,
            scale=1.3,
            thickness=3,
        )

    if not calibrated:
        elapsed = time.time() - calibration_start_time
        remaining = max(0, CALIBRATION_SECONDS - elapsed)
        put_text(frame, "Calibrating - keep eyes open", (20, 40), WARNING_COLOR)
        put_text(frame, f"Remaining: {remaining:.1f}s", (20, 75), WARNING_COLOR)
        return frame

    status_text = "Blink detected" if blink_detected else "No blink"
    status_color = SUCCESS_COLOR if blink_detected else TEXT_COLOR

    put_text(frame, status_text, (20, 40), status_color, scale=1)
    put_text(frame, f"EAR: {ear:.3f}", (20, 75), TEXT_COLOR)
    put_text(frame, f"Threshold: {threshold:.3f}", (20, 110), TEXT_COLOR)
    put_text(frame, f"Blink count: {blink_count}", (20, 145), TEXT_COLOR)

    return frame

def run_blink_detection():
    """Run the real-time blink detection loop."""
    camera = open_camera()
    face_mesh = create_face_mesh()

    calibration_start_time = time.time()
    calibration_values = []
    threshold = None
    open_eye_average = None
    consecutive_closed_frames = 0
    last_blink_time = 0
    blink_count = 0
    blink_detected = False

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Frame could not be read from the camera.")
                break

            face_landmarks_list = detect_face_landmarks(frame, face_mesh)
            face_detected = face_landmarks_list is not None
            ear = 0

            if face_detected:
                face_landmarks = face_landmarks_list[0]
                frame = draw_eye_contours(frame, face_landmarks)
                ear = calculate_average_ear(frame, face_landmarks)

                if threshold is None:
                    calibration_values.append(ear)
                    elapsed = time.time() - calibration_start_time

                    enough_time = elapsed >= CALIBRATION_SECONDS
                    enough_samples = len(calibration_values) >= MIN_CALIBRATION_SAMPLES

                    if enough_time and enough_samples:
                        threshold, open_eye_average = calibrate_threshold(
                            calibration_values
                        )
                        print(f"Calibration completed.")
                        print(f"Open-eye EAR average: {open_eye_average:.3f}")
                        print(f"Blink threshold: {threshold:.3f}")
                else:
                    (
                        blink_detected,
                        consecutive_closed_frames,
                        last_blink_time,
                        blink_count,
                    ) = update_blink_state(
                        ear,
                        threshold,
                        consecutive_closed_frames,
                        last_blink_time,
                        blink_count,
                    )
            else:
                blink_detected = False
                consecutive_closed_frames = 0

            calibrated = threshold is not None
            frame = draw_runtime_info(
                frame,
                face_detected,
                calibrated,
                blink_detected,
                ear,
                threshold or 0,
                blink_count,
                calibration_start_time,
            )

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        camera.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_blink_detection()