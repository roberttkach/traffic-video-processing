import cv2
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.layers.convolutional import Conv2D


model = Sequential()
model.add(Conv2D(32, (1, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='linear'))  # x, y координаты

model.compile(optimizer='adam', loss='mse')
cap = cv2.VideoCapture('/home/hallex/Документы/hattt/datasets/KRA-2-9-2023-08-22-evening.mp4')

tracker = cv2.TrackerKCF_create()

ret, frame = cap.read()
roi = cv2.selectROI(frame, False)
ret = tracker.init(frame, roi)

ret, frame = cap.read()
ret = tracker.init(frame, roi)

while True:
    # Чтение нового кадра
    ret, frame = cap.read()
    if not ret:
        break

    success, roi = tracker.update(frame)
    (x, y, w, h) = tuple(map(int, roi))

    
    cv2.imshow("Tracking", frame)

    # Выход из цикла, если нажата клавиша 'q'
    if cv2.waitKey(1) & 0xff == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
