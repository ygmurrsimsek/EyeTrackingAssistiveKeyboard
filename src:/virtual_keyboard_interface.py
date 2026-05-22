"""
Virtual keyboard interface with blink-based selection.

This module combines MediaPipe blink detection with a Tkinter virtual keyboard.
Letters and command buttons are highlighted automatically. When a blink is
detected, the active key is selected and the message area is updated.
"""

import time
import tkinter as tk
from tkinter import messagebox

import cv2
import mediapipe as mp
import numpy as np

try:
    import pyttsx3
except ImportError:
    pyttsx3 = None


CAMERA_INDEX = 0

CALIBRATION_SECONDS = 3
MIN_CALIBRATION_SAMPLES = 15
THRESHOLD_SCALE = 0.75
CONSECUTIVE_BLINK_FRAMES = 3
BLINK_COOLDOWN_SECONDS = 0.8

SCAN_INTERVAL_MS = 900
CAMERA_UPDATE_INTERVAL_MS = 15

BACKGROUND_COLOR = "#111111"
TEXT_COLOR = "#ffffff"
BUTTON_COLOR = "#222222"
ACTIVE_COLOR = "#ffd23f"
COMMAND_COLOR = "#1f4e79"
MESSAGE_COLOR = "#000000"

LEFT_EYE_EAR_POINTS = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_EAR_POINTS = [362, 385, 387, 263, 373, 380]

LETTERS = list("ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ")
COMMAND_KEYS = ["Boşluk", "Sil", "Temizle", "Oku"]
READY_MESSAGES = ["Yardım", "Su istiyorum", "Evet", "Hayır", "Ağrım var"]


def create_speech_engine():
    """Create an optional text-to-speech engine."""
    if pyttsx3 is None:
        return None

    try:
        return pyttsx3.init()
    except Exception:
        return None


def create_face_mesh():
    """Create a MediaPipe Face Mesh detector."""
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def landmark_to_pixel(landmark, frame_width, frame_height):
    """Convert a normalized MediaPipe landmark to pixel coordinates."""
    return np.array([int(landmark.x * frame_width), int(landmark.y * frame_height)])


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
    """Calculate the average eye openness value for both eyes."""
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


class VirtualKeyboardApp:
    """Tkinter app for blink-controlled virtual keyboard input."""

    def __init__(self, root):
        self.root = root
        self.root.title("Blink Controlled Virtual Keyboard")
        self.root.configure(bg=BACKGROUND_COLOR)
        self.root.bind("<q>", lambda event: self.close())

        self.camera = cv2.VideoCapture(CAMERA_INDEX)
        if not self.camera.isOpened():
            messagebox.showerror("Camera Error", "Camera could not be opened.")
            self.root.destroy()
            return

        self.face_mesh = create_face_mesh()
        self.speech_engine = create_speech_engine()

        self.all_keys = LETTERS + COMMAND_KEYS + READY_MESSAGES
        self.key_buttons = {}
        self.active_index = 0

        self.calibration_start_time = time.time()
        self.calibration_values = []
        self.threshold = None
        self.consecutive_closed_frames = 0
        self.last_blink_time = 0
        self.blink_count = 0

        self.message_var = tk.StringVar()
        self.status_var = tk.StringVar(value="Kalibrasyon: 3 saniye gözlerinizi açık tutun")

        self.build_interface()
        self.highlight_active_key()
        self.update_camera_loop()
        self.scan_next_key()

    def build_interface(self):
        """Build the message area, status area, and virtual keyboard."""
        title_label = tk.Label(
            self.root,
            text="Göz Kırpma ile Mesaj Yazma Sistemi",
            font=("Arial", 24, "bold"),
            bg=BACKGROUND_COLOR,
            fg=TEXT_COLOR,
        )
        title_label.pack(pady=12)

        message_entry = tk.Entry(
            self.root,
            textvariable=self.message_var,
            font=("Arial", 28, "bold"),
            bg=MESSAGE_COLOR,
            fg=TEXT_COLOR,
            insertbackground=TEXT_COLOR,
            justify="center",
            width=32,
        )
        message_entry.pack(padx=20, pady=12, ipady=12)

        status_label = tk.Label(
            self.root,
            textvariable=self.status_var,
            font=("Arial", 16, "bold"),
            bg=BACKGROUND_COLOR,
            fg=ACTIVE_COLOR,
        )
        status_label.pack(pady=8)

        keyboard_frame = tk.Frame(self.root, bg=BACKGROUND_COLOR)
        keyboard_frame.pack(padx=20, pady=12)

        self.create_keyboard_buttons(keyboard_frame)

    def create_keyboard_buttons(self, parent):
        """Create large high-contrast keyboard and command buttons."""
        rows = [
            list("ABCÇDEFGĞ"),
            list("HIİJKLMNOÖ"),
            list("PRSŞTUÜVYZ"),
            COMMAND_KEYS,
            READY_MESSAGES,
        ]

        for row_index, row_keys in enumerate(rows):
            row_frame = tk.Frame(parent, bg=BACKGROUND_COLOR)
            row_frame.grid(row=row_index, column=0, pady=5)

            for column_index, key in enumerate(row_keys):
                color = COMMAND_COLOR if key in COMMAND_KEYS + READY_MESSAGES else BUTTON_COLOR
                button = tk.Button(
                    row_frame,
                    text=key,
                    font=("Arial", 18, "bold"),
                    width=10 if len(key) > 1 else 4,
                    height=2,
                    bg=color,
                    fg=TEXT_COLOR,
                    activebackground=ACTIVE_COLOR,
                    activeforeground="#000000",
                    command=lambda selected_key=key: self.select_key(selected_key),
                )
                button.grid(row=0, column=column_index, padx=4, pady=4)
                self.key_buttons[key] = button

    def reset_button_colors(self):
        """Reset all key colors before highlighting the active key."""
        for key, button in self.key_buttons.items():
            color = COMMAND_COLOR if key in COMMAND_KEYS + READY_MESSAGES else BUTTON_COLOR
            button.configure(bg=color, fg=TEXT_COLOR)

    def highlight_active_key(self):
        """Highlight the currently active key."""
        self.reset_button_colors()
        active_key = self.all_keys[self.active_index]
        self.key_buttons[active_key].configure(bg=ACTIVE_COLOR, fg="#000000")

    def scan_next_key(self):
        """Move the active highlight to the next key automatically."""
        self.active_index = (self.active_index + 1) % len(self.all_keys)
        self.highlight_active_key()
        self.root.after(SCAN_INTERVAL_MS, self.scan_next_key)

    def select_active_key(self):
        """Select the currently highlighted key after blink detection."""
        active_key = self.all_keys[self.active_index]
        self.select_key(active_key)

    def select_key(self, key):
        """Apply the selected letter, command, or ready message."""
        current_message = self.message_var.get()

        if key == "Boşluk":
            self.message_var.set(current_message + " ")
        elif key == "Sil":
            self.message_var.set(current_message[:-1])
        elif key == "Temizle":
            self.message_var.set("")
        elif key == "Oku":
            self.speak_message()
        elif key in READY_MESSAGES:
            separator = " " if current_message else ""
            self.message_var.set(current_message + separator + key)
        else:
            self.message_var.set(current_message + key)

    def speak_message(self):
        """Read the current message aloud if text-to-speech is available."""
        message = self.message_var.get().strip()

        if not message:
            self.status_var.set("Okunacak mesaj yok")
            return

        if self.speech_engine is None:
            self.status_var.set("Sesli okuma için pyttsx3 kurulu değil")
            print(f"Message to read: {message}")
            return

        self.speech_engine.say(message)
        self.speech_engine.runAndWait()
        self.status_var.set("Mesaj okundu")

    def detect_face_landmarks(self, frame):
        """Detect face landmarks from the current camera frame."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        results = self.face_mesh.process(rgb_frame)
        return results.multi_face_landmarks

    def update_blink_state(self, ear):
        """Detect blink using threshold, consecutive frames, and cooldown."""
        current_time = time.time()

        if ear < self.threshold:
            self.consecutive_closed_frames += 1
        else:
            self.consecutive_closed_frames = 0

        enough_closed_frames = self.consecutive_closed_frames >= CONSECUTIVE_BLINK_FRAMES
        cooldown_finished = current_time - self.last_blink_time > BLINK_COOLDOWN_SECONDS

        if enough_closed_frames and cooldown_finished:
            self.blink_count += 1
            self.last_blink_time = current_time
            self.select_active_key()
            self.status_var.set(f"Blink detected - Blink count: {self.blink_count}")
            print(f"Blink count: {self.blink_count}")
        else:
            self.status_var.set(f"No blink - Blink count: {self.blink_count}")

    def handle_calibration(self, ear):
        """Collect open-eye EAR samples and create a user-specific threshold."""
        self.calibration_values.append(ear)
        elapsed = time.time() - self.calibration_start_time
        remaining = max(0, CALIBRATION_SECONDS - elapsed)
        self.status_var.set(f"Kalibrasyon: gözlerinizi açık tutun ({remaining:.1f}s)")

        enough_time = elapsed >= CALIBRATION_SECONDS
        enough_samples = len(self.calibration_values) >= MIN_CALIBRATION_SAMPLES

        if enough_time and enough_samples:
            open_eye_average = float(np.mean(self.calibration_values))
            self.threshold = open_eye_average * THRESHOLD_SCALE
            self.status_var.set(
                f"Kalibrasyon tamamlandı - Eşik: {self.threshold:.3f}"
            )
            print("Calibration completed.")
            print(f"Open-eye EAR average: {open_eye_average:.3f}")
            print(f"Blink threshold: {self.threshold:.3f}")

    def update_camera_loop(self):
        """Read camera frames and update blink detection continuously."""
        success, frame = self.camera.read()

        if success:
            face_landmarks_list = self.detect_face_landmarks(frame)

            if face_landmarks_list is None:
                self.consecutive_closed_frames = 0
                self.status_var.set("No face detected")
            else:
                face_landmarks = face_landmarks_list[0]
                ear = calculate_average_ear(frame, face_landmarks)

                if self.threshold is None:
                    self.handle_calibration(ear)
                else:
                    self.update_blink_state(ear)
        else:
            self.status_var.set("Camera frame could not be read")

        self.root.after(CAMERA_UPDATE_INTERVAL_MS, self.update_camera_loop)

    def close(self):
        """Release resources and close the application."""
        if self.camera is not None:
            self.camera.release()

        if self.face_mesh is not None:
            self.face_mesh.close()

        self.root.destroy()


def main():
    """Start the virtual keyboard application."""
    root = tk.Tk()
    app = VirtualKeyboardApp(root)
    root.protocol("WM_DELETE_WINDOW", app.close)
    root.mainloop()


if __name__ == "__main__":
    main()
