import cv2
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D


# Model definition
model = Sequential()
model.add(Conv2D(32, (1, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='linear'))  # Output layer for x, y coordinates
model.compile(optimizer='adam', loss='mse')

# Video capture
cap = cv2.VideoCapture('/home/hallex/Документы/hattt/datasets/KRA-2-9-2023-08-22-evening.mp4')
tracker = cv2.TrackerKCF_create()

# ROI selection
ret, frame = cap.read()
roi = cv2.selectROI(frame, False)
ret = tracker.init(frame, roi)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Tracker update
    success, roi = tracker.update(frame)
    (x, y, w, h) = tuple(map(int, roi))

    cv2.imshow("Tracking", frame)
    if cv2.waitKey(1) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
