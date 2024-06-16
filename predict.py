#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2
import os
from ultralytics import YOLO

n_frame_skip = 1  # Каждый n-ый кадр
visualize = False  # Переменная для управления визуализацией

# Загрузка модели YOLO
model = YOLO('model/weights/best.pt')  

def create_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Директории
input_videos_dir = 'data/videos'
input_images_dir = 'data/images'
create_dir(input_videos_dir)
create_dir(input_images_dir)

output_labels_videos_dir = 'output_labels/videos'
output_labels_images_dir = 'output_labels/images'
create_dir(output_labels_videos_dir)
create_dir(output_labels_images_dir)


def process_image(image_path, output_dir):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (640, 480))
    results = model.predict(image)
    boxes_ = results[0].cpu().boxes

    # Создаем имя файла для меток текущего изображения
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    txt_filename = os.path.join(output_dir, f"{image_filename}.txt")

    with open(txt_filename, 'w') as f:
        for i, box in enumerate(boxes_):
            cls = int(box.cls.numpy()[0])
            conf = box.conf.numpy()[0]

            if conf < 0.4 and cls != 0:
                continue
            if cls == 1 and conf < 0.6:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy.numpy()[0])
            x_center = (x1 + x2) / 2 / image.shape[1]
            y_center = (y1 + y2) / 2 / image.shape[0]
            width = (x2 - x1) / image.shape[1]
            height = (y2 - y1) / image.shape[0]

            f.write(f"{cls} {x_center} {y_center} {width} {height} {conf}\n")

            if visualize:
                label = f"{class_names[cls]}_{conf:.2f}"
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(image, label, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if visualize:
        cv2.imshow('YOLOv9 Object Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Обработка изображений
for image_filename in os.listdir(input_images_dir):
    image_path = os.path.join(input_images_dir, image_filename)
    process_image(image_path, output_labels_images_dir)

# Обработка видео
for video_filename in os.listdir(input_videos_dir):
    if video_filename.endswith('.mp4') or video_filename.endswith('.avi'):
        video_path = os.path.join(input_videos_dir, video_filename)
        output_dir = os.path.join(output_labels_videos_dir, os.path.splitext(video_filename)[0])
        create_dir(output_dir)

        cap = cv2.VideoCapture(video_path)
        class_names = {
            0: 'copter',
            1: 'airplane',
            2: 'helicopter',
            3: 'bird',
            4: 'aircraft'
        }

        if not cap.isOpened():
            print(f"Ошибка открытия видеофайла {video_filename}")
            continue

        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Ошибка захвата кадра из видеофайла {video_filename}")
                break

            frame = cv2.resize(frame, (640, 480))
            frame_count += 1

            if frame_count % n_frame_skip == 0:
                results = model.predict(frame)
                boxes_ = results[0].cpu().boxes

                # Создаем имя файла для меток текущего кадра
                txt_filename = os.path.join(output_dir, f"frame_{frame_count}.txt")

                with open(txt_filename, 'w') as f:
                    for i, box in enumerate(boxes_):
                        cls = int(box.cls.numpy()[0])
                        conf = box.conf.numpy()[0]

                        x1, y1, x2, y2 = map(int, box.xyxy.numpy()[0])
                        x_center = (x1 + x2) / 2 / frame.shape[1]
                        y_center = (y1 + y2) / 2 / frame.shape[0]
                        width = (x2 - x1) / frame.shape[1]
                        height = (y2 - y1) / frame.shape[0]

                        f.write(f"{cls} {x_center} {y_center} {width} {height} {conf}\n")

                        if visualize:
                            label = f"{class_names[cls]}_{conf:.2f}"
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label, (x1, y1 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if visualize:
                cv2.imshow('YOLOv9 Object Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        cap.release()
        cv2.destroyAllWindows()

