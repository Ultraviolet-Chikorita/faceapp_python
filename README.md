# FaceRecognitionApp

<p align="center">
  <strong>A Python + Kivy prototype for recognising known people from a live camera feed.</strong>
</p>

<p align="center">
  <a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white"></a>
  <a href="https://kivy.org/"><img alt="Kivy" src="https://img.shields.io/badge/UI-Kivy-5A5A5A"></a>
  <a href="https://opencv.org/"><img alt="OpenCV" src="https://img.shields.io/badge/Vision-OpenCV-5C3EE8?logo=opencv&logoColor=white"></a>
  <a href="https://github.com/ageitgey/face_recognition"><img alt="face_recognition" src="https://img.shields.io/badge/Face%20Recognition-dlib-orange"></a>
  <img alt="Android experimental" src="https://img.shields.io/badge/Android-experimental-3DDC84?logo=android&logoColor=white">
  <img alt="Status" src="https://img.shields.io/badge/status-prototype-yellow">
</p>

---

## Overview

**FaceRecognitionApp** is an experimental camera application built with **Kivy**, **OpenCV**, and [`face_recognition`](https://github.com/ageitgey/face_recognition).

At startup, the app identifies the current user via a phone-number lookup, requests a set of known people from a remote API, downloads their reference photos, generates face encodings, and then compares those encodings with faces seen by the device camera.

When a match is found, the app draws a bounding box and displays the person's **name** and **description** over the camera frame.

> [!NOTE]
> This repository is a prototype and is **not yet a production-ready biometric identity system**.

---

## What It Does

```mermaid
flowchart TD
    A["App starts"] --> B{"Running on Android?"}

    B -->|Yes| C["Attempt phone-number lookup<br/>via Android TelephonyManager"]
    B -->|No| D["Use development fallback"]

    C --> E["POST phone number to user API"]
    D --> E

    E --> F["Receive known people"]
    F --> G["Download reference images"]
    G --> H["Detect faces in each image"]
    H --> I["Generate face encodings"]
    I --> J["Store known-face encodings in memory"]

    A --> K["Open camera"]
    K --> L["Capture frames"]
    J --> M["Compare detected faces<br/>with known encodings"]
    L --> M

    M --> N{"Match?"}
    N -->|Yes| O["Draw bounding box"]
    O --> P["Display name + description"]
    N -->|No| Q["Continue scanning"]

    P --> L
    Q --> L
```

---

## Architecture

```mermaid
flowchart LR
    subgraph Device["Client device"]
        UI["Kivy UI"]
        Camera["Camera / OpenCV"]
        Recognizer["face_recognition + dlib"]
        Android["PyJNIus / Android APIs"]
    end

    subgraph Backend["Remote services"]
        API["Person lookup API"]
        Images["Reference image URLs"]
    end

    Android -->|"phone number"| API
    API -->|"user_info"| UI
    API -->|"image URLs"| Images
    Images -->|"JPEG / PNG bytes"| Recognizer
    Camera -->|"RGB frames"| Recognizer
    Recognizer -->|"matches + locations"| UI
    UI -->|"annotated texture"| Display["Screen"]
```

### Main responsibilities

| Component | Responsibility |
|---|---|
| `get_phone_number()` | Attempts to obtain the Android device phone number |
| `fetch_user_info()` | Requests the people associated with that number |
| `prepare_known_faces()` | Downloads images and generates face encodings |
| `FaceCameraWidget` | Captures frames, performs recognition, and renders overlays |
| `FaceRecognitionApp` | Coordinates startup and background face loading |

---

## Recognition Sequence

```mermaid
sequenceDiagram
    autonumber
    participant App as FaceRecognitionApp
    participant Android as Android APIs
    participant API as Remote API
    participant ImageHost as Image Host
    participant FR as face_recognition
    participant Camera as Camera

    App->>Android: Request device phone number
    Android-->>App: Phone number / unavailable

    App->>API: POST phone_number
    API-->>App: user_info[]

    loop For every known person image
        App->>ImageHost: GET image
        ImageHost-->>App: Image bytes
        App->>FR: Detect face + create encoding
        FR-->>App: Face encoding
    end

    App->>Camera: Start capture

    loop Camera processing
        Camera-->>App: Frame
        App->>FR: Detect and encode faces
        FR-->>App: Face locations + encodings
        App->>FR: Compare with known encodings
        FR-->>App: Match result
        App->>App: Draw label for recognised face
    end
```

---

## Data Model

The app expects the backend to return a `user_info` collection with data shaped approximately like this:

```json
{
  "user_info": [
    {
      "display_name": "Example Person",
      "description": "Example description",
      "images": [
        {
          "url": "https://example.com/reference-image.jpg"
        }
      ]
    }
  ]
}
```

Each usable reference image is converted into a face encoding and associated with that person's name and description.

```mermaid
erDiagram
    USER_LOOKUP ||--o{ PERSON : returns
    PERSON ||--o{ REFERENCE_IMAGE : has
    PERSON ||--o{ FACE_ENCODING : generates

    USER_LOOKUP {
        string phone_number
    }

    PERSON {
        string display_name
        string description
    }

    REFERENCE_IMAGE {
        string url
    }

    FACE_ENCODING {
        vector encoding
    }
```

---

## Repository Structure

```text
faceapp_python/
├── main.py             # Application and recognition pipeline
├── requirements.txt    # Desktop/development Python dependencies
├── buildozer.spec      # Experimental Android packaging configuration
├── README.md
└── readme.txt
```

---

## Tech Stack

| Area | Technology |
|---|---|
| Language | Python |
| UI | Kivy |
| Camera / image processing | OpenCV |
| Face detection / embeddings | `face_recognition` + dlib |
| Numerical processing | NumPy |
| HTTP | Requests |
| Android bridge | PyJNIus |
| Android packaging | Buildozer / python-for-android |

---

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/Ultraviolet-Chikorita/faceapp_python.git
cd faceapp_python
```

### 2. Create a virtual environment

#### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

#### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> [!WARNING]
> `requirements.txt` currently includes Windows-specific packages such as `pywin32`, `pypiwin32`, and Kivy Windows dependencies. It should not yet be treated as a fully cross-platform dependency file.

`dlib` can also require native build tools and CMake, depending on the platform and Python version.

### 4. Run the app

```bash
python main.py
```

The desktop prototype attempts to open camera device `0`.

---

## Android Build

The repository includes an experimental `buildozer.spec`.

The current package configuration uses:

```ini
title = FaceRecognitionApp
package.name = facerecognition
version = 0.1
orientation = portrait
fullscreen = 1
```

Declared Android permissions:

```text
INTERNET
CAMERA
READ_PHONE_STATE
```

A typical Buildozer command is:

```bash
buildozer android debug
```

> [!IMPORTANT]
> Android packaging is currently experimental. Native dependencies such as **dlib**, `face_recognition`, and OpenCV can require additional recipes or build changes under python-for-android.

### Android-specific startup

```mermaid
flowchart TD
    A["Android app launches"] --> B["PyJNIus loads Android classes"]
    B --> C["Access TelephonyManager"]
    C --> D["Attempt getLine1Number()"]
    D --> E{"Number available?"}
    E -->|Yes| F["Use device number for API lookup"]
    E -->|No| G["Use development fallback"]
    F --> H["Load known people"]
    G --> H
```

Phone-number retrieval is not guaranteed to work on every device, SIM, carrier, or Android version. Modern Android versions can impose additional restrictions beyond the permission declared in `buildozer.spec`.

---

## Recognition Details

The matching call currently uses a tolerance of:

```python
face_recognition.compare_faces(
    person["encodings"],
    face_encoding,
    tolerance=0.3,
)
```

A lower tolerance is stricter than the library's typical default and can reduce false positive matches, while potentially increasing false negatives.

### Reference-image preparation

For each remote image, the app:

1. downloads the image with `requests`;
2. decodes it with OpenCV;
3. converts BGR to RGB;
4. reduces very wide images to a maximum width of **600 px**;
5. locates faces;
6. creates an encoding for the first detected face;
7. stores that encoding against the person.

```mermaid
flowchart LR
    URL["Reference image URL"]
    Download["requests.get()"]
    Decode["cv2.imdecode()"]
    RGB["BGR → RGB"]
    Resize{"Width > 600?"}
    Scale["Resize image"]
    Detect["Detect face"]
    Encode["Generate encoding"]
    Store["Store in known_faces"]

    URL --> Download --> Decode --> RGB --> Resize
    Resize -->|Yes| Scale --> Detect
    Resize -->|No| Detect
    Detect --> Encode --> Store
```

---

## Frame Processing

The Kivy callback is scheduled at approximately **60 callbacks per second**, while the recognition block runs only when:

```python
self.frame_count % 30 == 0
```

So recognition is intentionally skipped on most captured frames.

> [!NOTE]
> In the current implementation, texture rendering also occurs inside that same conditional block. This means the displayed image is updated only on recognition frames rather than on every captured frame.

A useful future refactor would separate:

- **camera rendering** — every frame;
- **face detection / encoding** — periodically;
- **recognition state** — cached between detection runs.

---

## Concurrency

Known-face preparation happens in a daemon thread so downloading and encoding reference images does not block the initial Kivy UI creation.

```mermaid
flowchart TD
    Start["App.build()"] --> Widget["Create FaceCameraWidget"]
    Start --> Thread["Start daemon loader thread"]

    Thread --> Download["Download known images"]
    Download --> Encode["Generate face encodings"]
    Encode --> Update["Assign camera_widget.known_faces"]

    Widget --> Capture["Camera capture loop"]
    Update --> Match["Recognition can use loaded faces"]
    Capture --> Match
```

---

## API Behaviour

The app currently sends a `POST` request to the configured person-by-phone endpoint:

```json
{
  "phone_number": "+44..."
}
```

and expects `user_info` in the JSON response.

The API URL and fallback identity are currently defined directly in `main.py`.

For a production implementation, move configuration into environment variables or a dedicated configuration layer, for example:

```text
FACEAPP_API_URL=
FACEAPP_API_TOKEN=
```

---

## Current Limitations

| Area | Current behaviour | Suggested improvement |
|---|---|---|
| Configuration | API endpoint is hard-coded | Environment/config file |
| Identity | Development fallback phone number is hard-coded | Explicit login/device identity |
| Authentication | No API authentication is shown | Authenticated requests |
| HTTP | Requests have no explicit timeout | Timeouts + retry policy |
| Caching | Reference images/encodings rebuild at startup | Local encrypted cache |
| Matching | Fixed `0.3` tolerance | Configurable/calibrated threshold |
| Unknown people | No explicit `Unknown` label | Add unknown state |
| Reference images | First detected face is encoded | Validate one intended face per image |
| Camera | Device `0` is assumed | Camera selection/configuration |
| Rendering | Display updates inside recognition interval | Render continuously, recognise periodically |
| Android | Native ML dependencies may be difficult to package | Android-compatible inference stack |
| Permissions | Basic permission declaration only | Runtime permission handling |
| Logging | Extensive `print()` debugging | Structured logging |
| Tests | No automated tests | Unit + integration tests |

---

## Privacy & Security

Facial recognition processes biometric information and should be treated as sensitive functionality.

Before using this system beyond experimentation, consider:

- obtaining clear consent from people whose images are processed;
- documenting the purpose and lawful basis for biometric processing;
- authenticating and encrypting API traffic;
- limiting access to reference images;
- avoiding unnecessary retention of raw photographs;
- protecting derived face embeddings;
- implementing deletion and retention controls;
- removing development fallback identities;
- minimising collection of phone numbers and device identifiers;
- performing a privacy/security review before deployment.

> [!CAUTION]
> Do not treat the current prototype as an authentication or access-control system without substantial security, privacy, reliability, and anti-spoofing work.

---

## Suggested Production Architecture

```mermaid
flowchart TB
    subgraph Client["Mobile client"]
        Camera["Camera"]
        Detector["On-device face detector"]
        Encoder["On-device embedding model"]
        UI["Kivy / native UI"]
        Cache["Encrypted local cache"]
    end

    subgraph Service["Authenticated backend"]
        Auth["Authentication"]
        PersonAPI["Person service"]
        Media["Protected reference media"]
    end

    Camera --> Detector --> Encoder
    Encoder --> UI

    Auth --> PersonAPI
    PersonAPI --> Media
    PersonAPI -->|"authorised metadata / embeddings"| Cache
    Cache --> Encoder

    UI -->|"authenticated request"| Auth
```

Possible longer-term improvements include moving recognition entirely on-device, returning precomputed embeddings instead of raw reference images, and avoiding phone-number-based identity where possible.

---

## Development Roadmap

```mermaid
timeline
    title Potential roadmap
    section Prototype cleanup
        Configuration : Remove hard-coded endpoint and fallback identity
        Reliability : Add HTTP timeouts and validation
        UI : Separate rendering from recognition frequency
    section Engineering
        Structure : Split API, vision, and UI modules
        Quality : Add unit and integration tests
        Performance : Cache encodings and optimise frame processing
    section Android
        Permissions : Add runtime permission flow
        Packaging : Stabilise native dependency builds
        Device testing : Test multiple cameras and Android versions
    section Production readiness
        Security : Authentication and encrypted storage
        Privacy : Consent, retention, deletion controls
        ML safety : Threshold calibration and anti-spoofing
```

---

## Suggested Project Layout

As the prototype grows, the single-file application could be split into something like:

```text
faceapp_python/
├── app/
│   ├── api.py
│   ├── camera.py
│   ├── config.py
│   ├── recognition.py
│   └── ui.py
├── tests/
│   ├── test_api.py
│   └── test_recognition.py
├── main.py
├── requirements.txt
├── requirements-android.txt
├── buildozer.spec
└── README.md
```

---

## Contributing

This is currently an experimental project, but contributions can focus on:

- improving Android compatibility;
- separating application concerns;
- recognition performance;
- API robustness;
- test coverage;
- privacy and security hardening.

A typical contribution workflow:

```bash
git checkout -b feature/my-change
git add .
git commit -m "Describe the change"
git push origin feature/my-change
```

Then open a pull request against `main`.

---

## License

No licence is currently included in the repository.

If this project is intended to be open source, add a `LICENSE` file and update this section with the selected licence.

---

<p align="center">
  <sub>Experimental face-recognition prototype built with Python, Kivy, OpenCV, and dlib.</sub>
</p>
