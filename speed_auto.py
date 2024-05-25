import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense


# Define the Keras model
def create_model():
    """Creates and compiles a Keras model for predicting speed."""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1))  # Output layer for speed prediction
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

model = create_model()

# Initialize video capture
video_path = '/home/hallex/doc/hattt/datasets/KRA-2-9-2023-08-22-evening.mp4'
cap = cv2.VideoCapture(video_path)

# Check if the video opened successfully
if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

speeds = []  # List to store speed values
frames = []  # List to store frames for batch processing

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Preprocess the frame
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame_resized = cv2.resize(frame_gray, (64, 64))
        frame_normalized = frame_resized.astype('float32') / 255.0
        frame_expanded = np.expand_dims(frame_normalized, axis=-1)  # Add channel dimension
        frames.append(frame_expanded)

        # For demonstration, speed is set to 0 (replace with actual speed calculation if available)
        speed = 0
        speeds.append(speed)

        # Train the model in batches of 32 frames
        if len(frames) == 32:
            frames_array = np.array(frames)
            speeds_array = np.array(speeds)
            model.train_on_batch(frames_array, speeds_array)
            frames = []  # Clear frames list after training
            speeds = []  # Clear speeds list after training
    else:
        break

# Release the video capture and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
