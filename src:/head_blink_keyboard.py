"""
OpenCV overlay keyboard controlled by head movement and blink.

This version keeps the large camera view from blink_detection.py style:
- Eye landmarks remain visible.
- EAR, threshold, and blink count remain visible in the top-left corner.
- A virtual keyboard is drawn directly on the OpenCV camera frame.
- Head movement changes the active key; blink selects it.
"""

import time

import cv2
import mediapipe as mp
import numpy as np


CAMERA_INDEX = 0
WINDOW_NAME = "Head Blink Overlay Keyboard - Press Q to Quit"

CALIBRATION_SECONDS = 3
MIN_CALIBRATION_SAMPLES = 15
THRESHOLD_SCALE = 0.75
CONSECUTIVE_BLINK_FRAMES = 3
BLINK_COOLDOWN_SECONDS = 0.7
MOVE_COOLDOWN_SECONDS = 0.7
REPEAT_KEY_COOLDOWN_SECONDS = 1.0
SELECTION_DWELL_SECONDS = 0.8

RIGHT_THRESHOLD = 0.035
LEFT_THRESHOLD = 0.055
HEAD_MOVE_THRESHOLD_Y = 0.055

# Keep this True for natural mirror-like webcam control.
# If right/left feels reversed on your computer, change it to False.
MIRROR_CAMERA = True

SUCCESS_COLOR = (0, 255, 0)
WARNING_COLOR = (0, 255, 255)
ERROR_COLOR = (0, 0, 255)
TEXT_COLOR = (255, 255, 255)
KEY_COLOR = (35, 35, 35)
COMMAND_COLOR = (90, 55, 20)
ACTIVE_COLOR = (0, 220, 255)
MESSAGE_BG_COLOR = (20, 20, 20)

NOSE_TIP_INDEX = 1

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

# OpenCV's default text renderer is safest with ASCII labels.
KEYBOARD_ROWS = [
    list("ABCDEF"),
    list("GHIJKL"),
    list("MNOPRS"),
    list("TUVYZ"),
    ["SPACE", "DEL", "CLEAR", "REST"],
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
    return mp.solutions.face_mesh.FaceMesh(
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

    return (vertical_distance_1 + vertical_distance_2) / (2 * horizontal_distance)


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


def get_head_position(face_landmarks):
    """Return normalized nose-tip x/y position as head movement reference."""
    nose = face_landmarks.landmark[NOSE_TIP_INDEX]
    return np.array([nose.x, nose.y])


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


def draw_message_area(frame, message, last_selected_key):
    """Draw the typed message area at the top of the camera frame."""
    frame_height, frame_width, _ = frame.shape
    cv2.rectangle(frame, (10, 170), (frame_width - 10, 225), MESSAGE_BG_COLOR, -1)
    cv2.rectangle(frame, (10, 170), (frame_width - 10, 225), ACTIVE_COLOR, 2)

    shown_message = message[-42:] if len(message) > 42 else message
    put_text(
        frame,
        f"Message: {shown_message}",
        (25, 207),
        TEXT_COLOR,
        scale=0.9,
        thickness=2,
    )
    put_text(
        frame,
        f"Last: {last_selected_key or '-'}",
        (frame_width - 185, 207),
        WARNING_COLOR,
        scale=0.7,
        thickness=2,
    )
    return frame


def draw_runtime_info(
    frame,
    face_detected,
    calibrated,
    blink_detected,
    ear,
    threshold,
    blink_count,
    calibration_start_time,
    dx,
    selection_cooldown_remaining,
    selection_ready,
    dwell_remaining,
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
        put_text(frame, "Calibrating - keep head centered and eyes open", (20, 40), WARNING_COLOR)
        put_text(frame, f"Remaining: {remaining:.1f}s", (20, 75), WARNING_COLOR)
        return frame

    status_text = "Blink detected" if blink_detected else "No blink"
    status_color = SUCCESS_COLOR if blink_detected else TEXT_COLOR

    put_text(frame, status_text, (20, 40), status_color, scale=1)
    put_text(frame, f"EAR: {ear:.3f}", (20, 75), TEXT_COLOR)
    put_text(frame, f"Threshold: {threshold:.3f}", (20, 110), TEXT_COLOR)
    put_text(frame, f"Blink count: {blink_count}", (20, 145), TEXT_COLOR)
    put_text(frame, f"dx: {dx:.3f}", (260, 75), TEXT_COLOR)

    if selection_ready:
        put_text(frame, "Ready to select", (260, 40), SUCCESS_COLOR, scale=0.8)
    else:
        put_text(frame, f"Wait: {dwell_remaining:.1f}s", (260, 40), WARNING_COLOR, scale=0.8)

    if selection_cooldown_remaining > 0:
        put_text(
            frame,
            f"repeat cooldown: {selection_cooldown_remaining:.1f}s",
            (260, 110),
            WARNING_COLOR,
            scale=0.7,
            thickness=2,
        )

    return frame


def draw_pause_overlay(frame):
    """Draw a large PAUSED overlay."""
    frame_height, frame_width, _ = frame.shape
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame_width, frame_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, frame)

    text = "PAUSED"
    text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 2.2, 5)
    text_x = (frame_width - text_size[0]) // 2
    text_y = frame_height // 2
    put_text(frame, text, (text_x, text_y), WARNING_COLOR, scale=2.2, thickness=5)
    put_text(
        frame,
        "Blink again to continue",
        (text_x - 35, text_y + 55),
        TEXT_COLOR,
        scale=0.9,
        thickness=2,
    )
    return frame


def get_key_position(row, col):
    """Return the key string at the active row and column."""
    return KEYBOARD_ROWS[row][col]


def clamp_position(row, col):
    """Keep active row and column inside keyboard boundaries."""
    row = max(0, min(row, len(KEYBOARD_ROWS) - 1))
    col = max(0, min(col, len(KEYBOARD_ROWS[row]) - 1))
    return row, col


def draw_keyboard(frame, active_row, active_col):
    """Draw the virtual keyboard overlay at the bottom of the frame."""
    frame_height, frame_width, _ = frame.shape
    keyboard_top = int(frame_height * 0.58)
    keyboard_bottom = frame_height - 10
    keyboard_height = keyboard_bottom - keyboard_top
    row_count = len(KEYBOARD_ROWS)
    row_height = keyboard_height // row_count

    overlay = frame.copy()
    cv2.rectangle(
        overlay,
        (8, keyboard_top - 8),
        (frame_width - 8, keyboard_bottom),
        (0, 0, 0),
        -1,
    )
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    for row_index, row_keys in enumerate(KEYBOARD_ROWS):
        key_count = len(row_keys)
        key_gap = 8
        key_width = (frame_width - 20 - key_gap * (key_count - 1)) // key_count
        y1 = keyboard_top + row_index * row_height + 4
        y2 = y1 + row_height - 8

        for col_index, key in enumerate(row_keys):
            x1 = 10 + col_index * (key_width + key_gap)
            x2 = x1 + key_width

            is_active = row_index == active_row and col_index == active_col
            is_command = key in ["SPACE", "DEL", "CLEAR", "REST"]
            color = ACTIVE_COLOR if is_active else COMMAND_COLOR if is_command else KEY_COLOR
            text_color = (0, 0, 0) if is_active else TEXT_COLOR

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), TEXT_COLOR, 2)

            text_size, _ = cv2.getTextSize(key, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            text_x = x1 + (key_width - text_size[0]) // 2
            text_y = y1 + (y2 - y1 + text_size[1]) // 2
            put_text(frame, key, (text_x, text_y), text_color, scale=0.8, thickness=2)

    return frame


def get_head_direction(head_position, center_head_position):
    """Estimate head movement direction from calibrated center position."""
    if center_head_position is None:
        return None

    dx = head_position[0] - center_head_position[0]
    dy = head_position[1] - center_head_position[1]

    if abs(dx) > abs(dy):
        if dx > RIGHT_THRESHOLD:
            return "right"
        if dx < -LEFT_THRESHOLD:
            return "left"
    else:
        if dy > HEAD_MOVE_THRESHOLD_Y:
            return "down"
        if dy < -HEAD_MOVE_THRESHOLD_Y:
            return "up"

    return None


def move_selection(active_row, active_col, direction):
    """Move active key based on head direction."""
    if direction == "right":
        active_col += 1
    elif direction == "left":
        active_col -= 1
    elif direction == "up":
        active_row -= 1
    elif direction == "down":
        active_row += 1

    return clamp_position(active_row, active_col)


def update_blink_state(
    ear,
    threshold,
    consecutive_closed_frames,
    last_blink_time,
    blink_count,
):
    """Detect blink using thresholding, frame filtering, and cooldown."""
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


def apply_selected_key(message, key):
    """Apply selected keyboard key to the message."""
    if key == "SPACE":
        return message + " "
    if key == "DEL":
        return message[:-1]
    if key == "CLEAR":
        return ""
    if key == "REST":
        return message
    return message + key


def can_select_key(key, last_selected_key, last_selection_time, active_key_changed):
    """Prevent accidental immediate repeat of the same key."""
    if active_key_changed:
        return True

    if key != last_selected_key:
        return True

    elapsed = time.time() - last_selection_time
    return elapsed >= REPEAT_KEY_COOLDOWN_SECONDS


def get_selection_cooldown_remaining(key, last_selected_key, last_selection_time):
    """Return remaining cooldown time for repeated key selection."""
    if key != last_selected_key:
        return 0

    elapsed = time.time() - last_selection_time
    return max(0, REPEAT_KEY_COOLDOWN_SECONDS - elapsed)


def get_dwell_status(active_key_since):
    """Return whether the active key has been stable long enough to select."""
    elapsed = time.time() - active_key_since
    remaining = max(0, SELECTION_DWELL_SECONDS - elapsed)
    return elapsed >= SELECTION_DWELL_SECONDS, remaining


def run_overlay_keyboard():
    """Run the camera overlay keyboard application."""
    camera = open_camera()
    face_mesh = create_face_mesh()

    calibration_start_time = time.time()
    calibration_ear_values = []
    calibration_head_values = []
    threshold = None
    open_eye_average = None
    center_head_position = None

    consecutive_closed_frames = 0
    last_blink_time = 0
    last_move_time = 0
    blink_count = 0
    blink_detected = False
    dx = 0
    active_row = 0
    active_col = 0
    previous_active_row = active_row
    previous_active_col = active_col
    active_key_since = time.time()
    message = ""
    paused = False
    last_selected_key = None
    last_selection_time = 0

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Frame could not be read from the camera.")
                break

            if MIRROR_CAMERA:
                frame = cv2.flip(frame, 1)

            face_landmarks_list = detect_face_landmarks(frame, face_mesh)
            face_detected = face_landmarks_list is not None
            ear = 0

            if face_detected:
                face_landmarks = face_landmarks_list[0]
                frame = draw_eye_contours(frame, face_landmarks)
                ear = calculate_average_ear(frame, face_landmarks)
                head_position = get_head_position(face_landmarks)

                frame_height, frame_width, _ = frame.shape
                nose_px = (
                    int(head_position[0] * frame_width),
                    int(head_position[1] * frame_height),
                )
                cv2.circle(frame, nose_px, 7, ACTIVE_COLOR, -1)

                if threshold is None or center_head_position is None:
                    calibration_ear_values.append(ear)
                    calibration_head_values.append(head_position)
                    elapsed = time.time() - calibration_start_time

                    enough_time = elapsed >= CALIBRATION_SECONDS
                    enough_samples = len(calibration_ear_values) >= MIN_CALIBRATION_SAMPLES

                    if enough_time and enough_samples:
                        threshold, open_eye_average = calibrate_threshold(
                            calibration_ear_values
                        )
                        center_head_position = np.mean(
                            np.array(calibration_head_values),
                            axis=0,
                        )
                        print("Calibration completed.")
                        print(f"Open-eye EAR average: {open_eye_average:.3f}")
                        print(f"Blink threshold: {threshold:.3f}")
                        print(f"Center head position: {center_head_position}")
                else:
                    dx = head_position[0] - center_head_position[0]
                    direction = get_head_direction(head_position, center_head_position)
                    current_time = time.time()

                    if (
                        direction is not None
                        and current_time - last_move_time > MOVE_COOLDOWN_SECONDS
                        and not paused
                    ):
                        old_active_row = active_row
                        old_active_col = active_col
                        active_row, active_col = move_selection(
                            active_row,
                            active_col,
                            direction,
                        )
                        if active_row != old_active_row or active_col != old_active_col:
                            active_key_since = current_time
                        last_move_time = current_time
                        print(f"Head move: {direction}")

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

                    if blink_detected:
                        selected_key = get_key_position(active_row, active_col)
                        active_key_changed = (
                            active_row != previous_active_row
                            or active_col != previous_active_col
                        )
                        selection_ready, _ = get_dwell_status(active_key_since)

                        if paused:
                            paused = False
                            last_selected_key = "REST"
                            last_selection_time = time.time()
                            active_key_since = time.time()
                            print("Pause mode off")
                        elif not selection_ready:
                            print(f"Blink ignored, wait on active key: {selected_key}")
                        elif selected_key == "REST":
                            paused = True
                            last_selected_key = selected_key
                            last_selection_time = time.time()
                            print("Pause mode on")
                        elif can_select_key(
                            selected_key,
                            last_selected_key,
                            last_selection_time,
                            active_key_changed,
                        ):
                            message = apply_selected_key(message, selected_key)
                            last_selected_key = selected_key
                            last_selection_time = time.time()
                            previous_active_row = active_row
                            previous_active_col = active_col
                            print(f"Selected key: {selected_key}")
                        else:
                            print(f"Repeated key blocked: {selected_key}")
            else:
                blink_detected = False
                consecutive_closed_frames = 0

            calibrated = threshold is not None and center_head_position is not None
            active_key = get_key_position(active_row, active_col)
            cooldown_remaining = get_selection_cooldown_remaining(
                active_key,
                last_selected_key,
                last_selection_time,
            )
            selection_ready, dwell_remaining = get_dwell_status(active_key_since)
            if paused:
                selection_ready = False
                dwell_remaining = 0
            frame = draw_runtime_info(
                frame,
                face_detected,
                calibrated,
                blink_detected,
                ear,
                threshold or 0,
                blink_count,
                calibration_start_time,
                dx,
                cooldown_remaining,
                selection_ready,
                dwell_remaining,
            )
            frame = draw_message_area(frame, message, last_selected_key)
            frame = draw_keyboard(frame, active_row, active_col)

            if paused:
                frame = draw_pause_overlay(frame)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        camera.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_overlay_keyboard()
