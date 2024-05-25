import json
import os
import pathlib
import cv2
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import numpy as np




def getJson(video_path, directory):
    """Функция для извлечения данных из JSON файла."""
    filename = pathlib.Path(video_path).stem
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
        return json_file
    else:
        print(f'Файл {json_file} не найден')



def create_model():
    """Создание модели"""
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))

    return model

def compile_model(model):
    """Компиляция модели"""
    opt = Adam(learning_rate=0.001, decay=1e-6)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

def set_callbacks():
    """Настройка callbacks"""
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)
    checkpoint = ModelCheckpoint('best_model.h5', monitor='val_loss', save_best_only=True)

    return [early_stopping, checkpoint]

def train_model(model, x_train, y_train, x_val, y_val):
    """Обучение модели"""
    callbacks = set_callbacks()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=10, callbacks=callbacks)

    return history

def print_history(history):
    """Вывод истории обучения в консоли"""
    print("History of training:\n", history.history)

def process_video(video_path, json_directory):
    """Обработка видео"""
    json_data = getJson(video_path, json_directory)
    if json_data is None:
        return

    cap = cv2.VideoCapture(video_path)
    frames = []

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret == False:
            break
        frames.append(frame)

    cap.release()
    cv2.destroyAllWindows()

    processed_frames = []
    for frame in frames:
        areas = [[[int(x[0]*frame.shape[1]), int(x[1]*frame.shape[0])] for x in area] for area in json_data['areas']]
        zones = [[[int(x[0]*frame.shape[1]), int(x[1]*frame.shape[0])] for x in zone] for zone in json_data['zones']]

        trackers = [cv2.Tracker for _ in areas]
        for tracker, area in zip(trackers, areas):
            bbox = cv2.boundingRect(np.array(area))
            tracker.init(frame, bbox)

        for tracker, area in zip(trackers, areas):
            ret, bbox = tracker.update(frame)

            if ret:
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
            else:
                cv2.putText(frame, "Tracking failure detected", (100, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)

        # Зоны
        for zone in zones:
            cv2.polylines(frame, [np.array(zone)], True, (0, 255, 0), 2)

        frame = cv2.resize(frame, (128, 128))
        frame = image.img_to_array(frame)
        frame = frame/255
        processed_frames.append(frame)

    processed_frames = np.array(processed_frames)
    predictions = model.predict(processed_frames)

    return predictions

model = create_model()
compile_model(model)



json_data = getJson(r"C:\Users\diner\Desktop\KRA-2-7-2023-08-23-evening.mp4", r"D:\DATA\markup\jsons")
predictions = process_video(r"C:\Users\diner\Desktop\KRA-2-7-2023-08-23-evening.mp4")
print(predictions)
