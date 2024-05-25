from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense
import cv2
import numpy as np


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 1)))
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(1))

model.compile(optimizer='adam', loss='mean_squared_error')
cap = cv2.VideoCapture('/home/hallex/Документы/hattt/datasets/KRA-2-9-2023-08-22-evening.mp4')

speeds = []
frames = []

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        # Предобработка кадра
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, (64, 64))
        frame = frame.astype('float32') / 255.0
        frame = np.expand_dims(frame, axis=-1)  # Добавление измерения канала
        frames.append(frame)

        speed = 0
        speeds.append(speed)

        if len(frames) == 32:  # Обучение на каждых 32 кадрах
            model.train_on_batch(np.expand_dims(np.array(frames), axis=0), np.array(speeds))  # Измерения пакета
            frames = []
            speeds = []
    else:
        break

cap.release()
cv2.destroyAllWindows()
