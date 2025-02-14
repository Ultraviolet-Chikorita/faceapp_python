[app]
# (Application title and package details)
title = FaceRecognitionApp
package.name = facerecognition
package.domain = org.example
source.dir = .
source.include_exts = py,kv,png,jpg,atlas
version = 0.1

# List your requirements here. Note that opencv-python-headless is used instead of opencv-python.
requirements = python3,kivy,requests,numpy,pyjnius,opencv-python-headless,git+https://github.com/ageitgey/face_recognition_models,face_recognition

# Permissions required by your app:
android.permissions = INTERNET, CAMERA, READ_PHONE_STATE

# Orientation and fullscreen mode:
orientation = portrait
fullscreen = 1

# (Optional) Android-specific settings:
# icon.filename = %(source.dir)s/icon.png

[buildozer]
log_level = 2
warn_on_root = 1
