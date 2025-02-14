import cv2
import numpy as np
import requests
import face_recognition
import threading
from kivy.app import App
from kivy.uix.image import Image
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.utils import platform

# On Android, use pyjnius to access Android APIs.
if platform == 'android':
    from jnius import autoclass

# --------------------------
# Utility functions
# --------------------------

def get_phone_number():
    """
    Attempts to retrieve the device's phone number on Android.
    For non-Android platforms (or if any error occurs), returns "Unknown".
    """
    if platform == 'android':
        try:
            PythonActivity = autoclass('org.kivy.android.PythonActivity')
            Context = autoclass('android.content.Context')
            activity = PythonActivity.mActivity
            telephony_manager = activity.getSystemService(Context.TELEPHONY_SERVICE)
            phone = telephony_manager.getLine1Number()
            return phone if phone else "Unknown"
        except Exception as e:
            print("Error retrieving phone number:", e)
            return None
    else:
        return None

def fetch_user_info(phone_number = "+447393724610"):
    """
    Makes a POST request to the external API with the phone number.
    Expects a JSON response that includes a 'user_info' key.
    """
    if phone_number is None:
        phone_number = "+447393724610"
    url = "https://remind-glasses-web.onrender.com/api/person/by-phone/"  # <<<--- Replace with your actual API endpoint
    payload = {'phone_number': phone_number}
    try:
        response = requests.post(url, json=payload)
        data = response.json()
        return data.get('user_info', [])
    except Exception as e:
        print("Error fetching user info:", e)
        return []

def prepare_known_faces(user_info):
    """
    For each person in the user_info list, downloads their images,
    finds a face in each image (if present), computes a face encoding,
    and builds a list of known face data.
    
    Returns a list of dictionaries, each with:
       - 'display_name'
       - 'description'
       - 'encodings' (list of face encoding vectors)
    """
    known_faces = []
    print(f"[DEBUG] Starting prepare_known_faces with {len(user_info)} entries")
    for person in user_info:
        name = person.get('display_name', 'Unknown')
        description = person.get('description', '')
        encodings = []
        images = person.get('images', [])
        print(f"[DEBUG] Processing person: {name} with {len(images)} images")
        for image_data in images:
            img_url = image_data.get('url')
            print(f"[DEBUG] Downloading image from: {img_url}")
            try:
                resp = requests.get(img_url)
                if resp.status_code == 200:
                    image_bytes = resp.content
                    np_arr = np.frombuffer(image_bytes, np.uint8)
                    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    if img is not None:
                        print(f"[DEBUG] Successfully decoded image from: {img_url}")
                        # Convert BGR (OpenCV) to RGB (face_recognition)
                        try:
                            print(f"[DEBUG] Original image shape: {img.shape}, dtype: {img.dtype}")
                            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            print(f"[DEBUG] Converted image to RGB; new shape: {rgb_img.shape}")
                            
                            # Optional: Resize the image if it's very large, to reduce processing load:
                            max_width = 600
                            if rgb_img.shape[1] > max_width:
                                scale_factor = max_width / rgb_img.shape[1]
                                new_width = int(rgb_img.shape[1] * scale_factor)
                                new_height = int(rgb_img.shape[0] * scale_factor)
                                rgb_img = cv2.resize(rgb_img, (new_width, new_height))
                                print(f"[DEBUG] Resized image to: {rgb_img.shape}")

                            face_locations = face_recognition.face_locations(rgb_img)
                            print(f"[DEBUG] Found {len(face_locations)} face(s)")
                        except Exception as e:
                            print(f"[ERROR] Exception during face detection: {e}")
                        print(f"[DEBUG] Found {len(face_locations)} face(s) in image")
                        if face_locations:
                            encoding = face_recognition.face_encodings(rgb_img, face_locations)[0]
                            encodings.append(encoding)
                            print(f"[DEBUG] Added encoding for person: {name}")
                        else:
                            print(f"[DEBUG] No face found in image: {img_url}")
                    else:
                        print(f"[DEBUG] cv2.imdecode failed for image: {img_url}")
                else:
                    print(f"[DEBUG] Failed to download image from {img_url} with status code {resp.status_code}")
            except Exception as e:
                print(f"[ERROR] Exception processing image from URL {img_url}: {e}")
        if encodings:
            known_faces.append({
                'display_name': name,
                'description': description,
                'encodings': encodings
            })
            print(f"[DEBUG] Added person {name} with {len(encodings)} encoding(s)")
        else:
            print(f"[DEBUG] No valid encodings found for person: {name}")
    print(f"[DEBUG] Finished prepare_known_faces. Total known_faces: {len(known_faces)}")
    return known_faces

# --------------------------
# Kivy Widget for Camera + Overlays
# --------------------------

class FaceCameraWidget(Image):
    """
    A Kivy Image widget that uses OpenCV to grab camera frames,
    runs face detection/recognition, and overlays bounding boxes and labels.
    """
    def __init__(self, **kwargs):
        super(FaceCameraWidget, self).__init__(**kwargs)
        # Start video capture (0 is the default camera)
        self.capture = cv2.VideoCapture(0)
        if not self.capture.isOpened():
            print("Failed to open camera. Please ensure a webcam is connected and not in use.")
        # Schedule the update function at roughly 30 FPS.
        Clock.schedule_interval(self.update, 1.0 / 60)
        # This list will be updated later with known face data.
        self.known_faces = []
        self.frame_count = 0

    def update(self, dt):
        try:
            ret, frame = self.capture.read()
            if not ret:
                print("No frame received from camera.")
                return
            
            self.frame_count += 1
            if self.frame_count % 30 == 0:
                # Convert the frame from BGR to RGB
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Detect faces in the frame
                face_locations = face_recognition.face_locations(rgb_frame)
                face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

                # For each detected face, try to match with a known face.
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    for person in self.known_faces:
                        # Compare with each stored encoding for this person.
                        matches = face_recognition.compare_faces(person['encodings'], face_encoding, tolerance=0.3)
                        if True in matches:
                            name = person['display_name']
                            description = person['description']
                            # Draw a bounding box (green rectangle) around the face.
                            cv2.rectangle(rgb_frame, (left, top), (right, bottom), (0, 255, 0), 2)
                            # Annotate with the person's name and description.
                            label = f"{name}: {description}"
                            cv2.putText(rgb_frame, label, (left, top - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                            break  # Found a match; no need to check further.

                # Convert the annotated image to a texture so it can be displayed.
                buf = rgb_frame.tobytes()
                texture = Texture.create(size=(rgb_frame.shape[1], rgb_frame.shape[0]), colorfmt='rgb')
                texture.blit_buffer(buf, colorfmt='rgb', bufferfmt='ubyte')
                self.texture = texture
        except Exception as e:
            print("Error in update method:", e)

# --------------------------
# The Main App Class
# --------------------------

class FaceRecognitionApp(App):
    def build(self):
        # Retrieve and store the phone number (global variable can be used elsewhere).
        self.phone_number = get_phone_number()
        print("Phone Number:", self.phone_number)

        # Fetch user info from the external API.
        self.user_info = fetch_user_info(self.phone_number)
        print("User Info:", self.user_info)

        # Create the camera widget that will display the live feed.
        self.camera_widget = FaceCameraWidget()

        # Load the known faces in a separate thread (so as not to block the UI).
        threading.Thread(target=self.load_known_faces, daemon=True).start()

        return self.camera_widget

    def load_known_faces(self):
        """
        Downloads images, computes face encodings, and updates the
        camera widget with known faces for recognition.
        """
        try:
            known_faces = prepare_known_faces(self.user_info)
            self.camera_widget.known_faces = known_faces
            print("Loaded known faces:", known_faces)
        except Exception as e:
            print("Error loading known faces:", e)
    

# --------------------------
# Run the App
# --------------------------

if __name__ == '__main__':
    FaceRecognitionApp().run()
