import os
import json

import numpy as np
import torch
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import cv2


# Define a simple Keras model (though it seems not used later in the script)
def create_keras_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='linear'))
    model.compile(optimizer='adam', loss='mse')
    return model


keras_model = create_keras_model()

# Load the YOLO model (ensure that the correct weights file is used)
model = torch.hub.load('ultralytics/yolov5', 'custom', path='yolo8x.pt')
models = model('/home/hallex/doc/hattt/models/best.pt')

video_dir = r"/home/hallex/doc/hattt/datasets"
json_dir = r"/home/hallex/doc/hattt/json"


def get_json(video_path, directory):
    """Extract data from a JSON file associated with a video."""
    filename = os.path.basename(video_path).split('.')[0]
    json_file = os.path.join(directory, filename + '.json')

    if os.path.exists(json_file):
        with open(json_file, 'r') as f:
            data = json.load(f)
            if 'areas' in data:
                print('Areas:')
                print(json.dumps(data['areas'], indent=4))
            if 'zones' in data:
                print('Zones:')
                print(json.dumps(data['zones'], indent=4))
        return data
    else:
        print(f'File {json_file} not found')
        return None  # Return None if file is not found


# List all video files in the directory
video_files = [f for f in os.listdir(video_dir) if f.endswith('.mp4')]

for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)
    cap = cv2.VideoCapture(video_path)

    json_data = get_json(video_path, json_dir)

    cars = []
    frame_count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            # YOLO model detection
            results = model(frame)
            detections = results.xyxy[0].cpu().numpy()  # Get detections

            for detection in detections:
                x1, y1, x2, y2, confidence, class_id = detection
                if class_id == 2 and confidence > 0.5:  # Assuming class_id 2 corresponds to 'car'
                    for area in json_data['areas']:
                        if x1 > area['x'] and y1 > area['y'] and x2 < area['x'] + area['w'] and y2 < area['y'] + area[
                            'h']:
                            # Draw rectangle around the car
                            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cars.append((frame_count, (x1 + x2) / 2, (y1 + y2) / 2))

            cv2.imshow('Frame', frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

            frame_count += 1
        else:
            break

    cap.release()
    cv2.destroyAllWindows()

    # Calculate the speed of each car
    for i in range(1, len(cars)):
        frame1, x1, y1 = cars[i - 1]
        frame2, x2, y2 = cars[i]

        distance = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        frames = frame2 - frame1

        # Speed as distance per frame
        speed = distance / frames

        print(f'Car {i} speed: {speed:.2f} pixels per frame')
