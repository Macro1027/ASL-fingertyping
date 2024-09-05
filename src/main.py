import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import torch
import pyautogui
from model.model import CNN
from model.utils import test_transform, unnormalize
from src.constants import *

def load_model(model_path):
    """
    Load the trained model from the specified path.
    """
    model = CNN(in_features=3, out_features=len(LABELS))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()
    return model

def initialize_webcam():
    """
    Initialize the webcam.
    """
    return cv2.VideoCapture(0)

def detect_hand(detector, img):
    """
    Detect hands in the image.
    """
    hands, img = detector.findHands(img, draw=False)

    return hands, img

def preprocess_roi(img, bbox):
    """
    Preprocess the region of interest (ROI) from the image.
    """
    x, y, w, h = bbox

    # Calculate padding (20% of width and height)
    pad_w = int(0.2 * w)
    pad_h = int(0.2 * h)

    # Adjust the bounding box coordinates to include padding
    x = max(0, x - pad_w)
    y = max(0, y - pad_h)
    w = min(w + 2 * pad_w, img.shape[1] - x)
    h = min(h + 2 * pad_h, img.shape[0] - y)

    # Extract the region of interest (ROI)
    roi = img[y:y+h, x:x+w] / 255.0

    if roi.size == 0:
        return None

    roi = cv2.resize(roi, (200, 200))
    roi = test_transform(roi)

    img_to_write = np.array(unnormalize(roi).permute((1, 2, 0))) * 255
    print(img_to_write)
    cv2.imwrite("img.png", img_to_write)

    roi = torch.tensor(roi, dtype=torch.float32)
    roi_tensor = roi.unsqueeze(0)
    return roi_tensor

def predict_gesture(model, roi_tensor):
    """
    Predict the gesture from the ROI tensor.
    """
    with torch.no_grad():
        prediction = model(roi_tensor)
        class_id = torch.argmax(prediction).item()
        probs = torch.softmax(prediction, dim=1).flatten()
        return class_id, probs

def display_prediction(img, bbox, class_name, probs, class_id):
    """
    Display the prediction on the image.
    """
    x, y, w, h = bbox
    if probs[class_id] > 0.055:
        cv2.putText(img, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

def type_character(class_name):
    """
    Type the detected character on the screen.
    """
    pyautogui.typewrite(class_name)

def main():
    model_path = f"{MODEL_PATH}/sign_language_model.pth"
    model = load_model(model_path)
    cap = initialize_webcam()
    detector = HandDetector(detectionCon=0.8, maxHands=1)

    while True:
        success, img = cap.read()
        if not success:
            break

        hands, img = detect_hand(detector, img)

        if hands:
            hand = hands[0]
            bbox = hand["bbox"]  # Bounding box info x,y,w,h
            roi_tensor = preprocess_roi(img, bbox)

            if roi_tensor is not None:
                class_id, probs = predict_gesture(model, roi_tensor)
                class_name = LABELS[class_id]
                display_prediction(img, bbox, class_name, probs, class_id)
                type_character(class_name)

        cv2.imshow('Sign Language Detection', img)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
