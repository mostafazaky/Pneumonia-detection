import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO
from util import visualize, set_background

# Load YOLOv5 model
model_path = "/Users/mostafazaky/health-care/models/train/weights/best.pt"
model = YOLO(model_path)

# Set background image
set_background('./bg.png')

# Set title
st.title('Pneumonia detection')

# Set header
st.header('Please upload an image')

# Upload file
file = st.file_uploader('', type=['png', 'jpg', 'jpeg'])

# Process the uploaded image
if file:
    image = Image.open(file).convert('RGB')
    image_array = np.asarray(image)

    # Perform prediction
    results = model(image_array)

    threshold = 0.5

    # Extract predictions
    pred_boxes = results[0].boxes.xyxy  # Bounding boxes in [x1, y1, x2, y2] format
    pred_scores = results[0].boxes.conf  # Confidence scores
    pred_classes = results[0].boxes.cls  # Class labels

    # Filter predictions based on threshold
    filtered_boxes = []
    filtered_classes = []
    for i, score in enumerate(pred_scores):
        if score > threshold:
            box = pred_boxes[i].tolist()
            x1, y1, x2, y2 = [int(coord) for coord in box]
            filtered_boxes.append([x1, y1, x2, y2])
            filtered_classes.append(int(pred_classes[i]))

    # Visualize the results
    visualize(image, filtered_boxes, filtered_classes)

