
import os
import time
import pickle
import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize the camera
cap = cv2.VideoCapture(0)  # Change the camera index if needed
if not cap.isOpened():
    print("Error: Could not open camera.")
    raise SystemExit(1)

# Determine expected feature length from the model
def _expected_feature_length(trained_model, fallback=42):
    n = getattr(trained_model, "n_features_in_", None)
    if isinstance(n, (int, np.integer)) and n > 0:
        return int(n)
    return fallback

EXPECTED_FEATURES = _expected_feature_length(model)

def _normalize_feature_length(vec, expected_len):
    if expected_len <= 0:
        return vec
    if len(vec) < expected_len:
        return vec + [0] * (expected_len - len(vec))
    if len(vec) > expected_len:
        return vec[:expected_len]
    return vec

# Initialize MediaPipe Hand Landmarker (Tasks API)
MODEL_PATH = os.environ.get("HAND_LANDMARKER_MODEL", "./hand_landmarker.task")
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "Missing hand landmarker model. Download a MediaPipe Hand Landmarker "
        "model (.task) and place it at ./hand_landmarker.task, or set "
        "HAND_LANDMARKER_MODEL to its path."
    )

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
RunningMode = getattr(vision, "RunningMode", None) or getattr(vision, "VisionRunningMode", None)
if RunningMode is None:
    raise RuntimeError("Unsupported mediapipe.tasks vision running mode API.")

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

# Mapping of labels
labels_dict = {0: 'Yes', 1: 'No', 2: 'Hello', 3: 'I love you', 4: 'Thank you'}

with HandLandmarker.create_from_options(options) as landmarker:
    while True:
        # Read a frame from the camera
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Variables for hand landmarks and data
        data_aux = []
        x_ = []
        y_ = []

        # Get frame dimensions
        H, W, _ = frame.shape

        # Convert the frame to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process hand landmarks using MediaPipe Tasks
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = landmarker.detect_for_video(mp_image, int(time.time() * 1000))

        if results.hand_landmarks:
            for hand_landmarks in results.hand_landmarks:
                for lm in hand_landmarks:
                    x_.append(lm.x)
                    y_.append(lm.y)

                min_x = min(x_)
                min_y = min(y_)
                for lm in hand_landmarks:
                    data_aux.append(lm.x - min_x)
                    data_aux.append(lm.y - min_y)

                for lm in hand_landmarks:
                    cx = int(lm.x * W)
                    cy = int(lm.y * H)
                    cv2.circle(frame, (cx, cy), 2, (0, 255, 0), -1)

            if len(data_aux) > 0:
                data_aux = _normalize_feature_length(data_aux, EXPECTED_FEATURES)

                # Calculate bounding box coordinates
                x1 = int(min(x_) * W) - 10
                y1 = int(min(y_) * H) - 10
                x2 = int(max(x_) * W) - 10
                y2 = int(max(y_) * H) - 10

                # Predict using the trained model
                prediction = model.predict([np.asarray(data_aux)])
                predicted_character = labels_dict.get(int(prediction[0]), 'Unknown')

                # Display the result on the frame
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 0), 4)
                cv2.putText(
                    frame,
                    predicted_character,
                    (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.3,
                    (0, 0, 0),
                    3,
                    cv2.LINE_AA,
                )

        # Display the frame
        cv2.imshow('frame', frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the camera and close all windows
cap.release()
cv2.destroyAllWindows()
