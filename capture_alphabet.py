import os
import sys
import logging
import absl.logging

# -------------------------
# ✅ Suppress TensorFlow & Mediapipe logs
# -------------------------
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
absl.logging.set_verbosity(absl.logging.FATAL)
logging.getLogger('absl').setLevel(logging.FATAL)
absl.logging._warn_preinit_stderr = False

def suppress_stderr():
    devnull = os.open(os.devnull, os.O_WRONLY)
    sys.stderr.flush()
    os.dup2(devnull, sys.stderr.fileno())

suppress_stderr()

import cv2
import time
import mediapipe as mp

# -------------------------
# ✅ Settings
# -------------------------
SAVE_PATH = "dataset/alphabet"
IMG_SIZE = (64, 64)
TOTAL_IMAGES = 100

# -------------------------
# ✅ Mediapipe setup
# -------------------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

def capture_images(label):
    label = label.upper()
    label_folder = os.path.join(SAVE_PATH, label)
    if not os.path.exists(label_folder):
        os.makedirs(label_folder)
        print(f"[INFO] Created folder: {label_folder}")
        count = 0
    else:
        existing = len(os.listdir(label_folder))
        print(f"[INFO] Folder already exists with {existing} image(s).")
        print("[CHOICE] Enter 'R' to reset the folder or 'E' to exit without capturing.")
        while True:
            choice = input("Do you want to (R)eset or (E)xit? [R/E]: ").strip().lower()
            if choice == 'r':
                for file in os.listdir(label_folder):
                    os.remove(os.path.join(label_folder, file))
                print("[INFO] All previous images deleted.")
                count = 0
                break
            elif choice == 'e':
                print("[INFO] Capture aborted by user.")
                return
            else:
                print("[ERROR] Invalid choice. Enter 'R' to reset or 'E' to exit.")

    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("[ERROR] Could not open camera.")
        return

    cv2.namedWindow("Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera", 800, 600)
    cv2.moveWindow("Camera", 100, 100)

    print("[INFO] Starting capture in 3 seconds...")
    time.sleep(3)
    print("[INFO] Capturing... Show hand. Press 'q' to quit early.")

    while count < TOTAL_IMAGES:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                h, w, _ = frame.shape
                x_vals = [int(lm.x * w) for lm in hand_landmarks.landmark]
                y_vals = [int(lm.y * h) for lm in hand_landmarks.landmark]
                x_min = max(min(x_vals) - 40, 0)
                x_max = min(max(x_vals) + 40, w)
                y_min = max(min(y_vals) - 40, 0)
                y_max = min(max(y_vals) + 40, h)

                roi = frame[y_min:y_max, x_min:x_max]
                if roi.size == 0 or (x_max - x_min < 40 or y_max - y_min < 40):
                    continue

                resized = cv2.resize(roi, IMG_SIZE)
                file_path = os.path.join(label_folder, f"{label}_{count}.jpg")
                cv2.imwrite(file_path, resized)
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

if __name__ == "__main__":
    print("Enter the alphabet to capture (A-Z): ", end='', flush=True)
    letter = input().strip().upper()
    if letter.isalpha() and len(letter) == 1:
        capture_images(letter)
    else:
        print("[ERROR] Please enter a valid single alphabet character.")
