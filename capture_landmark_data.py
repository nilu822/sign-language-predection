# capture_landmark_data.py
### ✅ Updated `capture_landmark_data.py`
import os
import cv2
import time
import numpy as np
import mediapipe as mp
import pyttsx3
import threading

tts = pyttsx3.init()
tts.setProperty('rate', 160)
tts.setProperty('voice', tts.getProperty('voices')[1].id)
tts_lock = threading.Lock()

def speak(text):
    def _speak():
        with tts_lock:
            try:
                tts.say(text)
                tts.runAndWait()
            except RuntimeError:
                print("[TTS ERROR] TTS already speaking or failed.")
    threading.Thread(target=_speak).start()

def capture_gesture_data(word, mode="landmark"):
    print(f"[INFO] Capturing gesture for: {word} ({mode})")
    speak(f"Tracking hand for {word}. Move your hand naturally.")

    DATA_PATH = f'dataset_{mode}/word_{mode}/{word}'
    os.makedirs(DATA_PATH, exist_ok=True)

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] External camera (1) not accessible. Please check connection.")
        return

    cap.set(3, 960)
    cap.set(4, 480)

    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    sample_count = 0
    total_samples = 200

    while sample_count < total_samples:
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                landmark = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()
                if landmark.shape == (42,):
                    file_index = str(sample_count).zfill(3)
                    np.save(os.path.join(DATA_PATH, f"{file_index}.npy"), landmark)
                    sample_count += 1

    cap.release()
    speak(f"Capture session ended for {word}.")
    print(f"[INFO] Capture session ended for '{word}'.")














# fronted --- 1st changes 
# import os
# import cv2
# import time
# import numpy as np
# import mediapipe as mp
# import pyttsx3
# import threading

# # TTS setup
# tts = pyttsx3.init()
# tts.setProperty('rate', 160)
# tts.setProperty('voice', tts.getProperty('voices')[1].id)
# tts_lock = threading.Lock()

# def speak(text):
#     def _speak():
#         with tts_lock:
#             try:
#                 tts.say(text)
#                 tts.runAndWait()
#             except RuntimeError:
#                 print("[TTS ERROR] TTS already speaking or failed.")
#     threading.Thread(target=_speak).start()
    

# def capture_gesture_data(word, mode="landmark"):
#     print(f"[INFO] Capturing gesture for: {word} ({mode})")
#     speak(f"Capturing gesture for {word} in landmark mode")

#     DATA_PATH = f'dataset_{mode}/word_{mode}/{word}'
#     os.makedirs(DATA_PATH, exist_ok=True)

#     # Force external camera only
#     cap = cv2.VideoCapture(1)
#     if not cap.isOpened():
#         print("[ERROR] External camera (1) not accessible. Please check connection.")
#         return

#     cap.set(3, 960)
#     cap.set(4, 480)

#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(max_num_hands=1)
#     mp_draw = mp.solutions.drawing_utils

#     print("[INFO] Starting capture... Show the gesture.")
#     sample_count = 0
#     total_samples = 200

#     while sample_count < total_samples:
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         frame = cv2.flip(frame, 1)
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(img_rgb)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#                 landmark = np.array([[lm.x, lm.y] for lm in hand_landmarks.landmark]).flatten()

#                 if landmark.shape == (42,):
#                     file_index = str(sample_count).zfill(3)
#                     np.save(os.path.join(DATA_PATH, f"{file_index}.npy"), landmark)
#                     sample_count += 1

#         cv2.putText(frame, f"Capturing: {sample_count}/{total_samples}", (20, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#         cv2.imshow("Landmark Data Capture", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     speak(f"{word} gesture captured successfully in landmark mode")
#     print(f"[INFO] Gesture '{word}' captured successfully in {mode} mode.")






















# was using this -- backend code -- original 
# import os
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# import cv2
# import sys
# import numpy as np
# import mediapipe as mp
# import pyttsx3
# import time
# import logging
# import warnings
# import absl.logging

# # Suppress logs and warnings
# absl.logging.set_verbosity(absl.logging.FATAL)
# logging.getLogger('absl').setLevel(logging.FATAL)
# warnings.filterwarnings("ignore")

# def suppress_stderr():
#     devnull = os.open(os.devnull, os.O_WRONLY)
#     sys.stderr.flush()
#     os.dup2(devnull, sys.stderr.fileno())
# suppress_stderr()

# # === Audio setup ===
# tts = pyttsx3.init()
# tts.setProperty('rate', 160)
# tts.setProperty('voice', tts.getProperty('voices')[1].id)

# def speak(text):
#     tts.say(text)
#     tts.runAndWait()

# def beep():
#     try:
#         import winsound
#         winsound.Beep(1000, 300)
#     except:
#         pass

# # === MediaPipe setup ===
# mp_hands = mp.solutions.hands
# hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
# mp_drawing = mp.solutions.drawing_utils

# # === Folder setup ===
# BASE_PATH = 'dataset_landmark/word_landmark'
# os.makedirs(BASE_PATH, exist_ok=True)

# # === Class name input and reset logic ===
# while True:
#     print("\n[INFO]:- Enter the class name to capture (e.g., thank you): ", end="", flush=True)
#     word = input().strip().capitalize()
#     class_dir = os.path.join(BASE_PATH, word)
#     preview_folder = os.path.join(class_dir, "preview")

#     if os.path.exists(class_dir):
#         print(f"\n[INFO]:- Class '{word}' already exists...")

#         while True:
#             print("[CHOICE]:- Press [R] to reset (overwrite) or [E] to exit: ", end="", flush=True)
#             choice = input().strip().lower()

#             if choice == 'r':
#                 for file in os.listdir(class_dir):
#                     if file.endswith('.npy'):
#                         os.remove(os.path.join(class_dir, file))
#                 if os.path.exists(preview_folder):
#                     for img in os.listdir(preview_folder):
#                         os.remove(os.path.join(preview_folder, img))
#                 print(f"[INFO]:- Reset complete. Old samples for '{word}' deleted.\n")
#                 break
#             elif choice == 'e':
#                 print("[INFO] Exiting.....")
#                 exit()
#             else:
#                 print("[INFO]:- Invalid key. Please enter 'R' to reset or 'E' to exit.\n")
#         break
#     else:
#         os.makedirs(class_dir)
#         break

# os.makedirs(preview_folder, exist_ok=True)
# start_count = len([f for f in os.listdir(class_dir) if f.endswith(".npy")])
# target_count = 200

# # === Webcam setup (lower resolution for speed) ===
# cap = cv2.VideoCapture(1)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# print(f"[INFO]:- Capturing starts in 3 seconds ..... ")
# print(f"[INFO]:- Capturing {target_count} samples for class: {word}")
# speak(f"Tracking hand for class {word}. Move your hand naturally.")

# sample_interval = 0.1
# last_capture_time = time.time()
# prev_time = time.time()

# halfway_announced = False
# full_announced = False

# while start_count < target_count:
#     ret, frame = cap.read()
#     if not ret:
#         continue

#     frame = cv2.flip(frame, 1)
#     height, width, _ = frame.shape
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(rgb)

#     current_time = time.time()
#     fps = 1 / (current_time - prev_time)
#     prev_time = current_time

#     if results.multi_hand_landmarks:
#         hand_landmarks = results.multi_hand_landmarks[0]
#         mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#         x_vals = [lm.x for lm in hand_landmarks.landmark]
#         y_vals = [lm.y for lm in hand_landmarks.landmark]
#         min_x, max_x = min(x_vals), max(x_vals)
#         min_y, max_y = min(y_vals), max(y_vals)
#         padding = 0.05
#         box_x1 = int(max((min_x - padding) * width, 0))
#         box_y1 = int(max((min_y - padding) * height, 0))
#         box_x2 = int(min((max_x + padding) * width, width))
#         box_y2 = int(min((max_y + padding) * height, height))
#         cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)

#         if time.time() - last_capture_time >= sample_interval:
#             # Save landmarks
#             landmark_list = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]
#             np.save(os.path.join(class_dir, f"{start_count}.npy"), np.array(landmark_list))

#             # Save cropped hand region
#             hand_crop = frame[box_y1:box_y2, box_x1:box_x2]
#             preview_path = os.path.join(preview_folder, f"{start_count}.jpg")
#             cv2.imwrite(preview_path, hand_crop, [cv2.IMWRITE_JPEG_QUALITY, 85])

#             start_count += 1
#             last_capture_time = time.time()
#             print(f"[INFO]:- Captured sample {start_count}/{target_count}")

#             if not halfway_announced and start_count >= target_count // 2:
#                 halfway_announced = True
#                 speak("Halfway completed")

#             if not full_announced and start_count == target_count:
#                 full_announced = True
#                 speak("Capture complete")
#                 beep()

#     # === UI Overlay ===
#     cv2.putText(frame, f"Class: {word}", (10, 30),
#                 cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     cv2.putText(frame, f"Samples: {start_count}/{target_count}", (10, 65),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
#     cv2.putText(frame, f"FPS: {fps:.2f}", (10, 100),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#     cv2.putText(frame, "Tracking hand... | Q to quit", (10, 130),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

#     cv2.imshow("Hand Capture", frame)
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # === Cleanup ===
# cap.release()
# cv2.destroyAllWindows()
# speak(f"Capture session ended for {word}.")
# print(f"[INFO]:- Finished capturing samples for '{word}'")
