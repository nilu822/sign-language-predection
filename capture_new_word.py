# correct code ---- was using this earlier 
# --- capture_new_word.py ---
import os
import sys
import time
import cv2
import logging
import absl.logging
import pyttsx3
import numpy as np
import mediapipe as mp

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)

def suppress_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    sys.stderr.flush()
    os.dup2(devnull, sys.stderr.fileno())
suppress_stderr()

# TTS
tts = pyttsx3.init()
tts.setProperty('rate', 160)
tts.setProperty('voice', tts.getProperty('voices')[1].id)
def speak(text):
    tts.say(text)
    tts.runAndWait()

SAVE_PATH = "dataset/word"
IMG_SIZE = (96, 96)
TOTAL_IMAGES = 300

# Mediapipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

def skin_mask(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower = np.array([0, 48, 80], dtype=np.uint8)
    upper = np.array([20, 255, 255], dtype=np.uint8)
    skin = cv2.inRange(hsv, lower, upper)
    return cv2.bitwise_and(img, img, mask=skin)

def capture_images(label):
    folder = os.path.join(SAVE_PATH, label)
    if not os.path.exists(folder):
        os.makedirs(folder)
        print(f"[INFO] Created folder: {folder}")
    else:
        existing = len(os.listdir(folder))
        print(f"[INFO] '{label}' folder already exists with {existing} image(s).")
        while True:
            print("[CHOICE] Enter 'R' to reset the folder or 'E' to exit without capturing.", flush=True)
            choice = input("Do you want to (R)eset or (E)xit? [R/E]: ").strip().lower()
            if choice == 'r':
                for file in os.listdir(folder):
                    os.remove(os.path.join(folder, file))
                print("[INFO] Folder reset.")
                break
            elif choice == 'e':
                print("[INFO] Capture aborted by user.")
                return
            else:
                print("[ERROR] Invalid choice. Please enter 'R' or 'E'.")

    cap = cv2.VideoCapture(1)
    # cap.set(3, 640)       # 1st setting of camera resoltuion
    # cap.set(4, 480)
    
    # cap.set(3, 1280)      #2nd  setting of camera resoltuion
    # cap.set(4, 720)

    cap.set(3, 1020)
    cap.set(4, 400)



    if not cap.isOpened():
        print("[ERROR] Failed to open camera.")
        return

    print("[INFO] Starting capture in 3 seconds...")
    time.sleep(3)
    speak("Capturing in progress. Please make the required gesture.")

    count = 0
    while count < TOTAL_IMAGES:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]

                x_min = max(min(x_list) - 40, 0)
                x_max = min(max(x_list) + 40, w)
                y_min = max(min(y_list) - 40, 0)
                y_max = min(max(y_list) + 40, h)

                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size == 0:
                    continue

                # masked = skin_mask(roi)
                # resized = cv2.resize(masked, IMG_SIZE)
                
                resized = cv2.resize(roi, IMG_SIZE)

                
                img_path = os.path.join(folder, f"{label}_{count}.jpg")
                cv2.imwrite(img_path, resized)
                count += 1

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                cv2.putText(frame, f"{label}: {count}/{TOTAL_IMAGES}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                break
        else:
            cv2.putText(frame, "No hand detected", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("[INFO] Capture manually stopped.")
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"[DONE] Saved {count} images for label '{label}'")
    speak("Captured and saved successfully.")

if __name__ == "__main__":
    print("Enter the word/label to capture (e.g., Hello, ThankYou): ", end='', flush=True)
    word = input().strip()
    if word:
        capture_images(word.capitalize())
    else:
        print("[ERROR] Label cannot be empty.")
























# # capture_new_word.py
# import os
# import sys
# import cv2
# import time
# import pyttsx3
# import mediapipe as mp
# import absl.logging
# import logging

# # ✅ Suppress TensorFlow & Mediapipe logs
# # # -------------------------
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# absl.logging.set_verbosity(absl.logging.FATAL)

# logging.getLogger('absl').setLevel(logging.FATAL)
# absl.logging._warn_preinit_stderr = False

# def suppress_stderr():
#     devnull = os.open(os.devnull, os.O_WRONLY)
#     sys.stderr.flush()
#     os.dup2(devnull, sys.stderr.fileno())

# suppress_stderr()


# tts = pyttsx3.init()
# tts.setProperty('rate', 160)
# tts.setProperty('voice', tts.getProperty('voices')[1].id)

# def speak(text):
#     tts.say(text)
#     tts.runAndWait()

# SAVE_PATH = "dataset/word"
# IMG_SIZE = (96, 96)
# TOTAL_IMAGES = 100

# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# def capture_images(label):
#     label_folder = os.path.join(SAVE_PATH, label)
#     if not os.path.exists(label_folder):
#         os.makedirs(label_folder)
#         count = 0
#     else:
#         existing_images = len(os.listdir(label_folder))
#         print(f"[INFO] Folder exists with {existing_images} images.")
#         choice = input("Reset folder (R) or Exit (E)? [R/E]: ").strip().lower()
#         if choice == 'r':
#             for file in os.listdir(label_folder):
#                 os.remove(os.path.join(label_folder, file))
#             count = 0
#         else:
#             return

#     cap = cv2.VideoCapture(1)
#     cap.set(3, 640)
#     cap.set(4, 480)
#     time.sleep(3)
#     speak("Starting capture")

#     count = 0
#     while count < TOTAL_IMAGES:
#         ret, frame = cap.read()
#         if not ret:
#             break
#         frame = cv2.flip(frame, 1)
#         results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 h, w, _ = frame.shape
#                 x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
#                 y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]

#                 x_min = max(min(x_list) - 40, 0)
#                 x_max = min(max(x_list) + 40, w)
#                 y_min = max(min(y_list) - 40, 0)
#                 y_max = min(max(y_list) + 40, h)

#                 roi = frame[y_min:y_max, x_min:x_max]
#                 if roi.size == 0:
#                     continue

#                 resized = cv2.resize(roi, IMG_SIZE)
#                 img_path = os.path.join(label_folder, f"{label}_{count}.jpg")
#                 cv2.imwrite(img_path, resized)
#                 count += 1

#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
#                 cv2.putText(frame, f"{label}: {count}/{TOTAL_IMAGES}", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#                 break
#         else:
#             cv2.putText(frame, "No hand detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         cv2.imshow("Capture", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     speak("Gesture saved successfully.")

# if __name__ == "__main__":
#     word = input("Enter the word to capture (e.g., Hello): ").strip()
#     if word:
#         capture_images(word.capitalize())































# without sound  --- 
# import os
# import sys
# import logging
# import absl.logging

# # -------------------------
# # ✅ Suppress TensorFlow & Mediapipe logs
# # -------------------------
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# absl.logging.set_verbosity(absl.logging.FATAL)
# logging.getLogger('absl').setLevel(logging.FATAL)
# absl.logging._warn_preinit_stderr = False

# def suppress_stderr():
#     devnull = os.open(os.devnull, os.O_WRONLY)
#     sys.stderr.flush()
#     os.dup2(devnull, sys.stderr.fileno())

# suppress_stderr()

# import cv2
# import time
# import mediapipe as mp

# # -------------------------
# # ✅ Settings
# # -------------------------
# SAVE_PATH = "dataset/word"
# IMG_SIZE = (64, 64)
# TOTAL_IMAGES = 100

# # -------------------------
# # ✅ Mediapipe setup
# # -------------------------
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# def capture_images(label):
#     # Create or handle existing folder
#     label_folder = os.path.join(SAVE_PATH, label)
#     if not os.path.exists(label_folder):
#         os.makedirs(label_folder)
#         print(f"[INFO] Created folder: {label_folder}")
#         count = 0
#     else:
#         existing_images = len(os.listdir(label_folder))
#         print(f"[INFO] Folder already exists with {existing_images} image(s).")
#         print("[CHOICE] Enter 'R' to reset the folder and start over, or 'E' to exit without capturing.")

#         while True:
#             choice = input("Do you want to (R)eset or (E)xit? [R/E]: ").strip().lower()
#             if choice == 'r':
#                 for file in os.listdir(label_folder):
#                     os.remove(os.path.join(label_folder, file))
#                 print("[INFO] All previous images deleted.")
#                 count = 0
#                 break
#             elif choice == 'e':
#                 print("[INFO] Capture aborted by user.")
#                 return
#             else:
#                 print("[ERROR] Invalid choice. Please enter 'R' to reset or 'E' to exit.")

#     # -------------------------
#     # ✅ Initialize Camera
#     # -------------------------
#     cap = cv2.VideoCapture(1)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

#     if not cap.isOpened():
#         print("[ERROR] Could not open external camera.")
#         return

#     print("[INFO] Starting capture in 3 seconds...")
#     time.sleep(3)
#     print("[INFO] Capturing... Move hand into view. Press 'q' to quit early.")

#     cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
#     cv2.resizeWindow("Camera", 800, 600)
#     cv2.moveWindow("Camera", 100, 100)

#     # -------------------------
#     # ✅ Capture loop
#     # -------------------------
#     while count < TOTAL_IMAGES:
#         ret, frame = cap.read()
#         if not ret:
#             print("[ERROR] Failed to capture frame.")
#             break

#         frame = cv2.flip(frame, 1)
#         img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(img_rgb)

#         if results.multi_hand_landmarks:
#             for hand_landmarks in results.multi_hand_landmarks:
#                 h, w, _ = frame.shape
#                 x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
#                 y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]

#                 x_min = max(min(x_list) - 40, 0)
#                 x_max = min(max(x_list) + 40, w)
#                 y_min = max(min(y_list) - 40, 0)
#                 y_max = min(max(y_list) + 40, h)

#                 roi = frame[y_min:y_max, x_min:x_max]
#                 if roi.size == 0 or (x_max - x_min < 40 or y_max - y_min < 40):
#                     continue

#                 # Save ROI image
#                 resized = cv2.resize(roi, IMG_SIZE)
#                 img_path = os.path.join(label_folder, f"{label}_{count}.jpg")
#                 cv2.imwrite(img_path, resized)
#                 count += 1

#                 # Draw landmarks and label
#                 mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
#                 cv2.putText(frame, f"{label}: {count}/{TOTAL_IMAGES}", (10, 30),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
#                 break
#         else:
#             cv2.putText(frame, "No hand detected", (10, 30),
#                         cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#         cv2.imshow("Camera", frame)

#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             print("[INFO] Capture manually stopped.")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     print(f"[DONE] Saved {count} images for label '{label}'")

# if __name__ == "__main__":
#     print("Enter the word/label to capture (e.g., Hello, ThankYou): ", end='', flush=True)
#     word = input().strip()
#     if word:
#         capture_images(word.capitalize())
#     else:
#         print("[ERROR] Label cannot be empty.")



