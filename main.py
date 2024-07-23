import cv2
import cvzone
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import torch
from src.model import CNN
from src.utils import greyscale_transform
from src.constants import *

# Load the trained model
m_path = f"{MODEL_PATH}/sign_language_model.pth"
model = CNN()
model.load_state_dict(torch.load(m_path))
model.eval()

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Create a hand detector object
detector = HandDetector(detectionCon=0.8, maxHands=1)

while True:
    # Capture frame-by-frame
    success, img = cap.read()
    if not success:
        break

    # Find the hands and their landmarks
    hands, img = detector.findHands(img, draw=False)
    _, draw = detector.findHands(img, draw=True)

    # If a hand is detected
    if hands:
        hand = hands[0]
        bbox = hand["bbox"]  # Bounding box info x,y,w,h
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
            continue  # Skip if ROI is empty

        # Extract the region of interest (ROI)
        roi = cv2.resize(roi, (28, 28))
        roi = torch.tensor(roi, dtype=torch.float32).permute(2, 0, 1)

        roi_tensor = greyscale_transform(roi)
        roi_tensor = roi_tensor.unsqueeze(0)

        roi_np = roi_tensor.squeeze(dim=0).permute((1, 2, 0)).numpy()
        roi_np = (roi_np * 0.5 + 0.5) * 255

        cv2.imwrite("img.png", roi_np)

        # Predict the gesture
        with torch.no_grad():
            prediction = model(roi_tensor)
            class_id = torch.argmax(prediction).item()
            probs = torch.softmax(prediction, dim=1).flatten()
            class_name = LABELS[class_id]

        # Display the prediction
        if probs[class_id] > 0.055:
            cv2.putText(img, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Sign Language Detection', img)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()