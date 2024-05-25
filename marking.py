import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D


# Define the Keras model
def create_keras_model():
    """Creates and compiles a Keras model for predicting x, y coordinates."""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='linear'))  # Output layer for x, y coordinates
    model.compile(optimizer='adam', loss='mse')
    return model

model = create_keras_model()

# Initialize video capture from the specified file
video_path = '/home/hallex/doc/hattt/datasets/KRA-2-9-2023-08-22-evening.mp4'
cap = cv2.VideoCapture(video_path)

# Initialize the tracker
tracker = cv2.TrackerKCF_create()

# Read the first frame and set up the ROI for tracking
ret, frame = cap.read()
if not ret:
    print("Failed to read the video file.")
    cap.release()
    exit()

# Let the user select the ROI (Region of Interest) for tracking
roi = cv2.selectROI("Select ROI", frame, False, False)
cv2.destroyWindow("Select ROI")

# Initialize the tracker with the selected ROI
ret = tracker.init(frame, roi)
if not ret:
    print("Failed to initialize the tracker.")
    cap.release()
    exit()

# Start tracking loop
while True:
    # Read a new frame
    ret, frame = cap.read()
    if not ret:
        break

    # Update the tracker
    success, roi = tracker.update(frame)
    if success:
        # Draw a rectangle around the tracked object
        (x, y, w, h) = tuple(map(int, roi))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the frame with the tracking rectangle
    cv2.imshow("Tracking", frame)

    # Exit the loop if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
