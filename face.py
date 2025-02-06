import cv2
import os
import time
import threading
import numpy as np
import face_recognition
from queue import Queue
from concurrent.futures import ThreadPoolExecutor, as_completed

# URL видеопотока (RTSP)
rtsp_url = "rtsp://admin:Belprom1!@192.168.0.168:554/ch01/0"

# Коэффициент масштабирования для ускорения обработки (уменьшаем изображение)
SCALE_FACTOR = 0.15

class VideoStreamReader:
    """
    Асинхронное чтение кадров из VideoCapture в отдельном потоке, чтобы не блокировать главный цикл.
    """
    def __init__(self, gst_pipeline):
        self.stream = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
        if not self.stream.isOpened():
            raise ValueError(f"Не удалось открыть GStreamer-поток: {gst_pipeline}")

        # Пытаемся уменьшить задержку (не всегда работает)
        self.stream.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.ret = False
        self.frame = None
        self.stopped = False

        # Оптимизации OpenCV
        cv2.setUseOptimized(True)
        cv2.setNumThreads(6)  # число потоков под ваше железо

        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()

    def _update(self):
        while not self.stopped:
            ret, frame = self.stream.read()
            if not ret:
                # time.sleep(0.01)
                continue
            self.ret = ret
            self.frame = frame

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()

def load_known_faces(img_folder='img'):
    """
    Загружает изображения известных лиц из папки img_folder и вычисляет их face encodings.
    Использует многопоточность для ускорения.
    """
    known_face_encodings = []
    known_face_names = []

    x = 0

    def process_image(file_name):
        path = os.path.join(img_folder, file_name)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            name = os.path.splitext(file_name)[0]
            return name, encodings[0]
        else:
            print(f"[WARNING] Лицо не найдено на изображении {file_name}")
            return None

    # Собираем все файлы для обработки
    image_files = [f for f in os.listdir(img_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # Обрабатываем изображения в потоках
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_image, file_name) for file_name in image_files]
        for future in as_completed(futures):
            x += 1
            print(x)
            result = future.result()
            if result:
                name, encoding = result
                known_face_encodings.append(encoding)
                known_face_names.append(name)
                print(f"[INFO] Загрузили лицо: {name}")

    return known_face_encodings, known_face_names

def main():
    # Загружаем базу известных лиц
    known_face_encodings, known_face_names = load_known_faces('img')

    # Открываем видеопоток
    cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
    cap = VideoStreamReader(rtsp_url)

    # if not cap.isOpened():
    #     print("[ERROR] Не удалось открыть видеопоток")
    #     return

    print("[INFO] Запуск распознавания. Нажмите Q для выхода.")
    
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Уменьшаем размер кадра для ускорения обработки
        small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Поиск лиц на уменьшенном изображении
        face_locations_small = face_recognition.face_locations(rgb_small_frame)
        face_encodings_small = face_recognition.face_encodings(rgb_small_frame, face_locations_small)
        
        # Масштабируем координаты лиц обратно к оригинальному размеру кадра
        face_locations = []
        for (top, right, bottom, left) in face_locations_small:
            top = int(top / SCALE_FACTOR)
            right = int(right / SCALE_FACTOR)
            bottom = int(bottom / SCALE_FACTOR)
            left = int(left / SCALE_FACTOR)
            face_locations.append((top, right, bottom, left))
        
        # Распознавание каждого найденного лица
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings_small):
            # Сравнение с базой известных лиц
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            name = "Unknown"
            if face_distances.size > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            # Отрисовка рамки и имени на оригинальном изображении
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(frame, name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Отображение результата
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Завершаем работу
    # cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()