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
- Python 3.8+
- `opencv-python`
- `mediapipe`
- `scikit-learn`
- `numpy`

Install dependencies:
```bash
pip install opencv-python mediapipe scikit-learn numpy
```

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

## Future Improvements
- Add more classes and improve dataset balance
- Switch to a more robust classifier or deep model
- Improve inference stability with temporal smoothing
