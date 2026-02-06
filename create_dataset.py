

import os
import pickle

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

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
    running_mode=RunningMode.IMAGE,
    num_hands=2,
    min_hand_detection_confidence=0.3,
    min_hand_presence_confidence=0.3,
    min_tracking_confidence=0.3,
)

DATA_DIR = './data'

data = []
labels = []
with HandLandmarker.create_from_options(options) as landmarker:
    for dir_ in os.listdir(DATA_DIR):
        dir_path = os.path.join(DATA_DIR, dir_)
        if os.path.isdir(dir_path):  # Skip non-directory entries
            for img_path in os.listdir(dir_path):
                img_path_full = os.path.join(dir_path, img_path)
                if os.path.isfile(img_path_full) and img_path.endswith(('.jpg', '.jpeg', '.png')):
                    data_aux = []

                    x_ = []
                    y_ = []

                    img = cv2.imread(img_path_full)
                    if img is None:
                        continue
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

                    results = landmarker.detect(mp_image)
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

                        data.append(data_aux)
                        labels.append(dir_)

with open('data.pickle', 'wb') as f:
    pickle.dump({'data': data, 'labels': labels}, f)
