import os
import threading

import cv2
import face_recognition
import numpy as np
import requests
from kivy.app import App
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.uix.image import Image
from kivy.utils import platform

# On Android, use pyjnius to access Android APIs.
if platform == "android":
    from jnius import autoclass


DEFAULT_API_URL = "https://remind-glasses-web.onrender.com/api/person/by-phone/"
REQUEST_TIMEOUT_SECONDS = 15


def get_phone_number():
    """Return the device phone number when Android exposes it, otherwise ``None``."""
    if platform != "android":
        return None

    try:
        PythonActivity = autoclass("org.kivy.android.PythonActivity")
        Context = autoclass("android.content.Context")
        activity = PythonActivity.mActivity
        telephony_manager = activity.getSystemService(Context.TELEPHONY_SERVICE)
        phone = telephony_manager.getLine1Number()
        return phone or None
    except Exception as exc:
        print("Error retrieving phone number:", exc)
        return None


def fetch_user_info(phone_number=None):
    """Fetch known-person records for the current/development user.

    A development identity must be supplied explicitly through
    ``REMIND_DEV_PHONE_NUMBER``. Deployed/reference-service calls should also set
    ``REMIND_API_TOKEN``; the phone number is a lookup key, not an auth factor.
    """
    phone_number = phone_number or os.getenv("REMIND_DEV_PHONE_NUMBER")
    if not phone_number:
        print(
            "No phone number is available. Set REMIND_DEV_PHONE_NUMBER for "
            "desktop development or run on a supported Android device."
        )
        return []

    url = os.getenv("REMIND_API_URL", DEFAULT_API_URL)
    token = os.getenv("REMIND_API_TOKEN", "")
    headers = {"X-Remind-Api-Key": token} if token else {}

    try:
        response = requests.post(
            url,
            json={"phone_number": phone_number},
            headers=headers,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        data = response.json()
        user_info = data.get("user_info", [])
        return user_info if isinstance(user_info, list) else []
    except (requests.RequestException, ValueError) as exc:
        print("Error fetching user info:", exc)
        return []


def prepare_known_faces(user_info):
    """Download reference images and build face encodings for known people."""
    known_faces = []
    print(f"[DEBUG] Starting prepare_known_faces with {len(user_info)} entries")

    for person in user_info:
        name = person.get("display_name", "Unknown")
        description = person.get("description", "")
        encodings = []
        images = person.get("images", [])
        print(f"[DEBUG] Processing person: {name} with {len(images)} images")

        for image_data in images:
            img_url = image_data.get("url")
            if not img_url:
                continue

            print(f"[DEBUG] Downloading reference image for: {name}")
            try:
                resp = requests.get(img_url, timeout=REQUEST_TIMEOUT_SECONDS)
                resp.raise_for_status()
                np_arr = np.frombuffer(resp.content, np.uint8)
                img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                if img is None:
                    print(f"[DEBUG] Could not decode reference image for {name}")
                    continue

                rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                max_width = 600
                if rgb_img.shape[1] > max_width:
                    scale_factor = max_width / rgb_img.shape[1]
                    new_width = int(rgb_img.shape[1] * scale_factor)
                    new_height = int(rgb_img.shape[0] * scale_factor)
                    rgb_img = cv2.resize(rgb_img, (new_width, new_height))

                face_locations = face_recognition.face_locations(rgb_img)
                print(f"[DEBUG] Found {len(face_locations)} face(s) for {name}")
                if face_locations:
                    encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
                    encodings.append(encoding)
            except (requests.RequestException, ValueError, cv2.error) as exc:
                print(f"[ERROR] Could not process reference image for {name}: {exc}")
            except Exception as exc:
                # Third-party face-recognition/dlib failures are surfaced per image so
                # one bad reference does not abort every known person.
                print(f"[ERROR] Face processing failed for {name}: {exc}")

        if encodings:
            known_faces.append(
                {
                    "display_name": name,
                    "description": description,
                    "encodings": encodings,
                }
            )
            print(f"[DEBUG] Added person {name} with {len(encodings)} encoding(s)")
        else:
            print(f"[DEBUG] No valid encodings found for person: {name}")

    print(f"[DEBUG] Finished prepare_known_faces. Total known_faces: {len(known_faces)}")
    return known_faces


class FaceCameraWidget(Image):
    """Kivy widget that recognises known faces in a live OpenCV camera feed."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Failed to open camera. Please ensure a webcam is connected and not in use.")
        Clock.schedule_interval(self.update, 1.0 / 60)
        self.known_faces = []
        self.frame_count = 0

    def update(self, _dt):
        try:
            ret, frame = self.capture.read()
            if not ret:
                return

            self.frame_count += 1
            if self.frame_count % 30 != 0:
                return

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(
                face_locations, face_encodings
            ):
                for person in self.known_faces:
                    matches = face_recognition.compare_faces(
                        person["encodings"], face_encoding, tolerance=0.3
                    )
                    if True not in matches:
                        continue

                    name = person["display_name"]
                    description = person["description"]
                    cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                    label = f"{name}: {description}"
                    cv2.putText(
                        rgb_frame,
                        label,
                        (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )
                    break

            buf = rgb_frame.tobytes()
            texture = Texture.create(
                size=(rgb_frame.shape[1], rgb_frame.shape[0]), colorfmt="rgb"
            )
            texture.blit_buffer(buf, colorfmt="rgb", bufferfmt="ubyte")
            self.texture = texture
        except Exception as exc:
            print("Error in camera update:", exc)

    def close(self):
        if self.capture is not None:
            self.capture.release()


class FaceRecognitionApp(App):
    def build(self):
        self.phone_number = get_phone_number()
        self.user_info = fetch_user_info(self.phone_number)
        self.camera_widget = FaceCameraWidget()
        threading.Thread(target=self.load_known_faces, daemon=True).start()
        return self.camera_widget

    def load_known_faces(self):
        try:
            known_faces = prepare_known_faces(self.user_info)
            self.camera_widget.known_faces = known_faces
            print(f"Loaded {len(known_faces)} known people")
        except Exception as exc:
            print("Error loading known faces:", exc)

    def on_stop(self):
        if hasattr(self, "camera_widget"):
            self.camera_widget.close()


if __name__ == "__main__":
    FaceRecognitionApp().run()
