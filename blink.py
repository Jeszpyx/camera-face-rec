import cv2
import os
import time
import threading
import numpy as np
import face_recognition
from queue import Queue

rtsp_url = "rtsp://admin:Belprom1!@192.168.0.168:554/ch01/0"
# ------------------ ПАРАМЕТРЫ ------------------

# Диапазон высоты лица (в пикселях), соответствующий ~1 м
IGNORE_RANGE_1M_MIN = 100
IGNORE_RANGE_1M_MAX = 130

# Порог для определения "закрытых" глаз
EYE_AR_THRESHOLD = 0.2

# Коэффициент масштабирования для ускорения face_recognition (например, 0.25)
SCALE_FACTOR = 0.15

# Период кулдауна для повторного распознавания одного и того же лица (в секундах)
COOLDOWN_PERIOD = 15

# ------------------ АСИНХРОННОЕ ЧТЕНИЕ ------------------

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
                time.sleep(0.01)
                continue
            self.ret = ret
            self.frame = frame

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.thread.join()
        self.stream.release()

# ------------------ ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ------------------

def load_known_faces(img_folder='img'):
    """
    Загружаем изображения из папки img_folder
    и вычисляем face encodings для каждого файла.
    """
    known_face_encodings = []
    known_face_names = []
    
    for file_name in os.listdir(img_folder):
        if file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            path = os.path.join(img_folder, file_name)
            image = face_recognition.load_image_file(path)
            encodings = face_recognition.face_encodings(image)
            if len(encodings) > 0:
                name = os.path.splitext(file_name)[0]
                known_face_encodings.append(encodings[0])
                known_face_names.append(name)
                print(f"[INFO] Загрузили лицо: {name}")
            else:
                print(f"[WARNING] Лицо не найдено на изображении {file_name}")
    return known_face_encodings, known_face_names

def eye_bounding_box_ratio(eye_points):
    xs = [p[0] for p in eye_points]
    ys = [p[1] for p in eye_points]
    w = max(xs) - min(xs)
    h = max(ys) - min(ys)
    if w < 1e-6:
        return 0.0
    return h / w

def are_eyes_closed(landmarks):
    left_eye = landmarks.get("left_eye")
    right_eye = landmarks.get("right_eye")
    if not (left_eye and right_eye):
        return False
    
    left_ratio = eye_bounding_box_ratio(left_eye)
    right_ratio = eye_bounding_box_ratio(right_eye)
    avg_ratio = (left_ratio + right_ratio) / 2.0
    
    return (avg_ratio < EYE_AR_THRESHOLD)

def send_open_door_request(name):
    # Здесь разместите вызов открытия двери или иной логики
    print(f"[ACTION] Дверь открыта для {name}")

# ------------------ ОБРАБОТКА РАСПОЗНАВАНИЯ В ОТДЕЛЬНОМ ПОТОКЕ ------------------

def recognition_worker(task_queue):
    """
    Функция-воркер, которая обрабатывает задачи по распознаванию лиц.
    При получении задачи вызывается send_open_door_request.
    """
    while True:
        task = task_queue.get()
        if task is None:
            break  # сигнал завершения
        name = task.get("name")
        send_open_door_request(name)
        task_queue.task_done()

# ------------------ ОСНОВНАЯ ФУНКЦИЯ ------------------

def main():
    # 1) Загрузка известных лиц
    known_face_encodings, known_face_names = load_known_faces('img')

    # 2) Подключение к потоку (асинхронное чтение кадров)
    try:
        cap = VideoStreamReader(rtsp_url)
    except ValueError as e:
        print("[ERROR]", e)
        return

    print("[INFO] Запуск распознавания. Нажмите Q для выхода.")

    # Очередь для задач распознавания (например, открытия двери)
    task_queue = Queue()
    worker_thread = threading.Thread(target=recognition_worker, args=(task_queue,), daemon=True)
    worker_thread.start()

    # Отслеживание состояний для каждого лица:
    # Для каждого имени будем хранить:
    #  - blinked: зафиксировано мигание
    #  - eye_state_prev: предыдущее состояние глаз (открыты/закрыты)
    #  - door_sent: запрос уже отправлен
    #  - last_recognized: время последнего успешного распознавания
    blink_states = {}

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            time.sleep(0.01)
            continue

        # Уменьшаем кадр для ускорения обработки
        small_frame = cv2.resize(frame, (0, 0), fx=SCALE_FACTOR, fy=SCALE_FACTOR)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        # Находим лица и вычисляем их параметры
        face_locations_small = face_recognition.face_locations(rgb_small_frame)
        face_encodings_small = face_recognition.face_encodings(rgb_small_frame, face_locations_small)
        face_landmarks_list_small = face_recognition.face_landmarks(rgb_small_frame, face_locations_small)

        # Масштабируем координаты обратно
        face_locations = []
        for (top, right, bottom, left) in face_locations_small:
            top = int(top / SCALE_FACTOR)
            right = int(right / SCALE_FACTOR)
            bottom = int(bottom / SCALE_FACTOR)
            left = int(left / SCALE_FACTOR)
            face_locations.append((top, right, bottom, left))

        current_time = time.time()
        # Обходим найденные лица
        for (top, right, bottom, left), face_encoding, landmarks in zip(face_locations, face_encodings_small, face_landmarks_list_small):
            face_height = bottom - top

            # Фильтр «1м зона»
            if IGNORE_RANGE_1M_MIN <= face_height <= IGNORE_RANGE_1M_MAX:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Too close", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                continue

            # Сравнение с базой лиц
            matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
            face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
            name = "Unknown"
            if len(face_distances) > 0:
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

            # Если лицо не найдено – пропускаем
            if name == "Unknown":
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                continue

            # Инициализация состояния для данного имени, если ранее не встречалось
            if name not in blink_states:
                blink_states[name] = {
                    "blinked": False,
                    "eye_state_prev": False,
                    "door_sent": False,
                    "last_recognized": 0
                }

            user_state = blink_states[name]

            # Если с момента последнего успешного распознавания прошло меньше COOLDOWN_PERIOD, пропускаем дальнейшую обработку
            if user_state["door_sent"] and (current_time - user_state["last_recognized"] < COOLDOWN_PERIOD):
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{name} (Cooldown)", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                continue
            elif user_state["door_sent"] and (current_time - user_state["last_recognized"] >= COOLDOWN_PERIOD):
                # Сброс состояния после кулдауна, чтобы лицо можно было обработать повторно
                user_state["blinked"] = False
                user_state["door_sent"] = False

            # Отслеживаем мигание
            eyes_now_closed = are_eyes_closed(landmarks)
            eyes_were_closed = user_state["eye_state_prev"]

            # Если произошёл переход: глаза закрылись, затем открылись – считаем, что был миг
            if (not eyes_now_closed) and eyes_were_closed:
                user_state["blinked"] = True

            user_state["eye_state_prev"] = eyes_now_closed

            if not user_state["blinked"]:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 165, 255), 2)
                cv2.putText(frame, f"{name} (Blink please!)", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 165, 255), 2)
            else:
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, name, (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                # Логируем событие
                print(f"[INFO] {name} обнаружен и мигнул (face_height={face_height})")
                if not user_state["door_sent"]:
                    # Отправляем задачу в очередь (обработка в отдельном потоке)
                    task_queue.put({"name": name})
                    user_state["door_sent"] = True
                    user_state["last_recognized"] = current_time

        # Отображение кадра
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Завершаем работу
    cap.stop()
    cv2.destroyAllWindows()
    # Посылаем сигнал завершения воркеру
    task_queue.put(None)
    worker_thread.join()

if __name__ == "__main__":
    main()