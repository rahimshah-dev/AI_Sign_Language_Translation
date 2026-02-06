# AI Sign Language Translation

Real-time hand-gesture recognition with MediaPipe, OpenCV, and a Random Forest
classifier. This project collects hand images from a webcam, extracts landmark
features, trains a lightweight model, and runs live inference to predict common
signs.

## Features
- Webcam data collection per class label
- Hand landmark feature extraction with MediaPipe
- Random Forest classifier training and evaluation
- Live inference with on-screen predictions

## Project Structure
- `collect_imgs.py` - Capture images per class using your webcam.
- `create_dataset.py` - Extract hand landmark features into `data.pickle`.
- `train_classifier.py` - Train the model and save `model.p`.
- `inference_classifier.py` - Run real-time predictions from the webcam.
- `data/` - Collected images (created after running `collect_imgs.py`).

## Requirements
- Python 3.12
- `opencv-python`
- `mediapipe==0.10.32` (Tasks API)
- `scikit-learn==1.8.0`
- `numpy`

Install dependencies:
```bash
pip install -r requirements.txt
```

## Model File (MediaPipe Tasks)
Download a MediaPipe Hand Landmarker model file (`.task`) and place it at:
```
./hand_landmarker.task
```
Or set the environment variable `HAND_LANDMARKER_MODEL` to the file path.

## Usage
1) Collect images for each class:
```bash
python collect_imgs.py
```
By default it creates 5 classes with 100 images each. Press `q` to start
capturing for each class.

2) Build the dataset:
```bash
python create_dataset.py
```
This writes `data.pickle` with features and labels.

3) Train the classifier:
```bash
python train_classifier.py
```
This writes `model.p` and prints accuracy on a test split.

4) Run inference:
```bash
python inference_classifier.py
```
Press `q` to quit the live window.

## Frontend
A deploy-ready landing page lives in `frontend/`.
Open `frontend/index.html` in a browser or host it on a static provider.
Update the GitHub link inside `frontend/index.html` before deploying.

## Browser-Only Inference (No Backend)
The frontend runs MediaPipe in the browser and uses a lightweight centroid
classifier stored in `frontend/model.json`.

If you retrain or add classes:
```bash
python3 tools/export_centroids.py
```

The MediaPipe model file is already included at:
```
frontend/hand_landmarker.task
```

## Vercel Deployment
This repo deploys a static frontend on Vercel. The Python inference pipeline
does not run in the browser, so keep it for local execution or a separate
backend service.

Steps:
1) Import the repo into Vercel.
2) Deploy (the `vercel.json` rewrite points `/` to `frontend/index.html`).

## Northflank Deployment (Always-On)
This setup uses a lightweight static server in Docker.

1) Create a new Northflank project and service.
2) Choose "Build from repository" and select this repo.
3) Set build type to Dockerfile.
4) Deploy. The service will serve the frontend from `/`.

## Labels
Default label mapping used in `inference_classifier.py`:
```
0: Yes
1: No
2: Hello
3: I love you
4: Thank you
```

## Notes
- If your webcam is not at index `0`, change the camera index in
  `collect_imgs.py` and `inference_classifier.py`.
- The current inference script pads features to match the trained model shape.
  If you change the feature extraction pipeline, retrain the model.
- If you see scikit-learn version mismatch warnings when loading `model.p`,
  retrain the model with `train_classifier.py` in your current environment.

## Future Improvements
- Add more classes and improve dataset balance
- Switch to a more robust classifier or deep model
- Improve inference stability with temporal smoothing
