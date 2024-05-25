import cv2
import json
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D
import os
import torch

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(64, 64, 3), activation='relu'))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(2, activation='linear'))
model.compile(optimizer='adam', loss='mse')

model = torch.hub.load('yolo8x.pt')
models = model('/home/hallex/Документы/hattt/models/best.pt')

videoo = r"/home/hallex/Документы/hattt/datasets"
jsons = r"/home/hallex/Документы/hattt/puk"



def getJson(video_path, directory):
    """Функция для извлечения данных из JSON файла."""
    import os
    import json

    filename = os.path.basename(video_path).split('.')[0]
    print('\n', filename,'\n')
    json_file = os.path.join(directory, filename + '.json')
    print('\n', json_file,'\n')

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
        print(f'Файл {json_file} не найден')
        return None  # возвращаем None, если файл не найден
    
files = os.listdir(videoo)

for filename in files:
    if filename.endswith('.mp4'):
        video = os.path.join(files, filename)        
        cap = cv2.VideoCapture(videoo) 

cars = []
frame_count = 0

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        detections = model.predict(frame)
        
        for detection in detections:
            x, y, w, h, confidence, class_id = detection

        # обнаружение по разметке
            if class_id == 'car' and confidence > 0.5:
                for area in json['areas']:
                    if x > area['x'] and y > area['y'] and x + w < area['x'] + area['w'] and y + h < area['y'] + area['h']:
                        #рисует прчмоугольник вокруг авто
                        cv2.retangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cars.append((frame_count, x, y))
                        
                        

        cv2.imshow('Frame', frame)

        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

        frame_count += 1
    else:
        break

cap.release()
cv2.destroyAllWindows()


#скорость каждого авто
for i in range(1, len(cars)):
    frame1, x1, y1 = cars[i - 1]
    frame2, x2, y2 = cars[i]
    
    distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    frames = frame2 - frame1
    
    #скорость как расстояние на кадр
    speed = distance / frames

    print(f'Car {i} speed: {speed} pixels per frame')
