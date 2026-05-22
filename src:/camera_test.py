"""
Camera test module for the eye tracking project.

This script opens the webcam, displays the live camera feed,
shows the current FPS value on the frame, and exits when the
user presses the Q key.
"""

import time

import cv2


CAMERA_INDEX = 0
WINDOW_NAME = "Camera Test - Press Q to Quit"


def open_camera(camera_index=CAMERA_INDEX):
    """Open the selected webcam and return the capture object."""
    camera = cv2.VideoCapture(camera_index)

    if not camera.isOpened():
        raise RuntimeError(
            "Camera could not be opened. Check webcam connection or camera index."
        )

    return camera


def calculate_fps(previous_time):
    """Calculate FPS using the time difference between two frames."""
    current_time = time.time()
    elapsed_time = current_time - previous_time

    if elapsed_time == 0:
        return 0, current_time

    fps = 1 / elapsed_time
    return fps, current_time


def draw_fps(frame, fps):
    """Draw the FPS value on the top-left corner of the frame."""
    fps_text = f"FPS: {fps:.2f}"

    cv2.putText(
        frame,
        fps_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )

    return frame


def run_camera_test():
    """Run the real-time webcam preview loop."""
    camera = open_camera()
    previous_time = time.time()

    try:
        while True:
            success, frame = camera.read()

            if not success:
                print("Frame could not be read from the camera.")
                break

            fps, previous_time = calculate_fps(previous_time)
            frame = draw_fps(frame, fps)

            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    run_camera_test()