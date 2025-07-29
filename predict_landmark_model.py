# === predict_landmark_model.py ===
# === predict_landmark_model.py ===

import os, time, cv2, numpy as np, pyttsx3, winsound, mediapipe as mp, threading
from tensorflow.keras.models import load_model
from collections import deque
from absl import logging
import tensorflow as tf

# Suppress logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.set_verbosity(logging.FATAL)
tf.get_logger().setLevel('ERROR')

# Load model and labels
model = load_model('model_landmark/sign_model_landmark.keras')
with open('model_landmark/labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# MediaPipe setup
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# TTS setup
engine = pyttsx3.init()
engine.setProperty('rate', 120)
engine.setProperty('volume', 1.0)
engine.setProperty('voice', engine.getProperty('voices')[1].id)
engine_lock = threading.Lock()
speak_thread_running = False



def reset_text():
    global typed_text, history, prev_prediction, consistency_counter
    global flash_timer, recent_prediction_time, spoken_once
    global speak_thread_running, engine, last_spoken_text
    typed_text, history, prev_prediction = "", [], ""
    consistency_counter = flash_timer = 0
    recent_prediction_time = time.time()
    spoken_once = False
    speak_thread_running = False
    last_spoken_text = ""
    try:
        if hasattr(engine, '_inLoop') and engine._inLoop:
            engine.endLoop()
        engine.stop()
    except Exception:
        pass
    engine = pyttsx3.init()
    engine.setProperty('rate', 120)
    engine.setProperty('volume', 1.0)
    engine.setProperty('voice', engine.getProperty('voices')[1].id)



def speak_async(text):
    global speak_thread_running
    if speak_thread_running:
        return  #original
    def speak():
        global speak_thread_running
        print(f"[DEBUG] Inside speak_async thread, about to speak: {text}")
        with engine_lock:
            try:
                speak_thread_running = True
                engine.say(text)
                engine.runAndWait()
                
                # speak_thread_running=False
                # spoken_once = True
            except RuntimeError as e:
                print(e)
                # pass
            finally:
                speak_thread_running = False
    
    threading.Thread(target=speak, daemon=True).start()
    # reset_text()

# Shared state
typed_text, history, prev_prediction = "", [], ""
prediction_buffer = deque(maxlen=10)
consistency_counter = flash_timer = prediction_count = 0
cooldown, required_consistency, repeat_delay = 12, 3, 2
recent_prediction_time = last_hand_seen = time.time()
spoken_once = False
last_spoken_text = ""

CAM_WIDTH, CAM_HEIGHT = 960, 480

def extract_landmarks(landmarks, frame_shape):
    h, w = frame_shape
    return np.array([[lm.x, lm.y] for lm in landmarks.landmark]).flatten()


def backspace_text():
    global typed_text, history, spoken_once, last_spoken_text
    if history:
        history.pop()
        typed_text = " ".join(history) + " " 
        # typed_text = " ".join(history) + " " + (" " if history else "") 
        if not history or typed_text.strip() == "":
            spoken_once = False
            last_spoken_text = ""
    else:
        # typed_text = ""   #original
        spoken_once = False
        last_spoken_text = ""

def get_current_text():
    return typed_text

def predict_landmark(frame):
    global typed_text, history, prev_prediction, spoken_once
    global prediction_buffer, consistency_counter, flash_timer
    global prediction_count, recent_prediction_time, last_hand_seen, last_spoken_text

    frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    current_time = time.time()
    label, confidence = "", 0
    unknown_gesture = False

    if results.multi_hand_landmarks:
        last_hand_seen = current_time
        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_list = [lm.x * w for lm in hand_landmarks.landmark]
            y_list = [lm.y * h for lm in hand_landmarks.landmark]
            x_min, x_max = max(0, int(min(x_list)) - 80), min(w, int(max(x_list)) + 80)
            y_min, y_max = max(0, int(min(y_list)) - 80), min(h, int(max(y_list)) + 80)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            if flash_timer > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), -1)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                flash_timer -= 1
            else:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

            landmark_input = extract_landmarks(hand_landmarks, (h, w))
            if landmark_input.shape != (42,):
                continue
            roi_input = np.expand_dims(landmark_input, axis=0)

            if prediction_count == 0:
                prediction = model.predict(roi_input, verbose=0)[0]
                confidence = np.max(prediction)
                label = labels[np.argmax(prediction)]
                prediction_buffer.append(label)

                if confidence > 0.85:
                    if label == prev_prediction:
                        consistency_counter += 1
                    else:
                        consistency_counter = 1
                        prev_prediction = label

                    if consistency_counter >= required_consistency:
                        if current_time - recent_prediction_time > repeat_delay:
                            old_text = typed_text.strip()
                            typed_text += label + " "
                            history.append(label)
                            recent_prediction_time = current_time
                            flash_timer = 5
                            winsound.Beep(800, 100)
                            if typed_text.strip() != old_text:
                                spoken_once = False
                                last_spoken_text = ""  # ✅ Reset speech tracker
                        consistency_counter = 0
                        unknown_gesture = False
                else:
                    prev_prediction = ""
                    consistency_counter = 0
                    unknown_gesture = True
                cv2.putText(frame, f"Prediction: {label} ({confidence:.2f})", (200, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    else:
        cv2.putText(frame, "No hand detected", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    if unknown_gesture:
        cv2.putText(frame, "❌ No known gesture found", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Display text
    words = typed_text.strip().split()
    line, y_pos = "", 360
    for word in words:
        test_line = line + word + " "
        (text_width, _), _ = cv2.getTextSize(test_line.strip(), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        if text_width < CAM_WIDTH:
            line = test_line
        else:
            cv2.putText(frame, line.strip(), (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            line = word + " "
            y_pos += 35
    if line:
        cv2.putText(frame, line.strip(), (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "[Waiting for prediction...]", (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    # ✅ SPEAK after 5s idle
    if (current_time - last_hand_seen) > 5 and typed_text.strip() and typed_text.strip() != last_spoken_text:
    # if (current_time - last_hand_seen) > 5 and typed_text.strip() and not spoken_once:
        print(f"[INFO] Speaking updated sentence: '{typed_text.strip()}'")
        print(speak_thread_running)
        speak_async(typed_text.strip())
        last_spoken_text = typed_text.strip()
        # spoken_once=True

    prediction_count = (prediction_count + 1) % cooldown
    return frame, typed_text.strip()
















# # predict_landmark_model.py
# # === predict_landmark_model.py ===     

# import os
# import time
# import cv2
# import numpy as np
# import tensorflow as tf
# import pyttsx3
# import winsound
# import mediapipe as mp
# from collections import deque
# from tensorflow.keras.models import load_model
# from absl import logging
# import threading

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# logging.set_verbosity(logging.FATAL)
# tf.get_logger().setLevel('ERROR')

# model = load_model('model_landmark/sign_model_landmark.keras')
# with open('model_landmark/labels.txt', 'r') as f:
#     labels = [line.strip() for line in f.readlines()]

# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# engine = pyttsx3.init()
# engine.setProperty('rate', 120)
# engine.setProperty('volume', 1.0)
# engine.setProperty('voice', engine.getProperty('voices')[1].id)

# engine_lock = threading.Lock()
# speak_thread_running = False

# def speak_async(text):
#     global speak_thread_running
#     if speak_thread_running:
#         return
#     speak_thread_running = True
#     def speak():
#         global speak_thread_running
#         with engine_lock:
#             try:
#                 # speak_thread_running = True
#                 engine.say(text)
#                 engine.runAndWait()
#             except RuntimeError as e:
#                 print(f"[TTS ERROR] {e}")
#             finally:
#                 speak_thread_running = False
    
#     threading.Thread(target=speak, daemon=True).start()

# # === Shared text state ===
# typed_text = ""
# history = []
# prev_prediction = ""
# prediction_buffer = deque(maxlen=10)
# consistency_counter = 0
# flash_timer = 0
# prediction_count = 0
# cooldown = 12
# required_consistency = 3
# repeat_delay = 2 # seconds
# recent_prediction_time = time.time()
# last_hand_seen = time.time()
# spoken_once = False

# CAM_WIDTH = 960
# CAM_HEIGHT = 480

# def extract_landmarks(landmarks, frame_shape):
#     h, w = frame_shape
#     return np.array([[lm.x, lm.y] for lm in landmarks.landmark]).flatten()

# def reset_text():
#     global typed_text, history, prev_prediction, consistency_counter, flash_timer, recent_prediction_time, spoken_once, speak_thread_running, engine
#     print("[LANDMARK DEBUG] reset_text() called - performing immediate reset")
#     typed_text = ""
#     history = []
#     prev_prediction = ""
#     consistency_counter = 0
#     flash_timer = 0
#     recent_prediction_time = time.time()
#     spoken_once = False # Crucial: Reset this flag to allow new speech after clearing text
#     speak_thread_running = False # Ensure the TTS thread is no longer considered "running"
    
#     # --- ADDED FOR ROBUSTNESS ---
#     try:
#         # Properly end the event loop if the engine is running
#         if hasattr(engine, '_inLoop') and engine._inLoop:
#             engine.endLoop()
#         engine.stop() # Stop the existing engine
#     except Exception as e:
#         print(f"[TTS ERROR] Error stopping TTS engine during reset: {e}")
    
#     # Re-initialize the TTS engine to ensure a fresh state
#     engine = pyttsx3.init()
#     engine.setProperty('rate', 120)
#     engine.setProperty('volume', 1.0)
#     engine.setProperty('voice', engine.getProperty('voices')[1].id)
#     # --- END ADDED ---

# def backspace_text():
#     global typed_text, history, spoken_once
#     print("[LANDMARK DEBUG] backspace_text() called - performing immediate backspace")
#     if history:
#         history.pop()
#         typed_text = " ".join(history) + " "
#         # If the last word was removed, or if history becomes empty, allow speaking again
#         if not history or typed_text.strip() == "":
#             spoken_once = False
#     else:
#         typed_text = ""
#         spoken_once = False # If text is empty, reset spoken_once for next predictions


# def get_current_text():
#     return typed_text 

# def predict_landmark(frame):
#     global typed_text, history, prev_prediction, spoken_once
#     global prediction_buffer, consistency_counter, flash_timer
#     global prediction_count, recent_prediction_time, last_hand_seen

#     frame = cv2.resize(frame, (CAM_WIDTH, CAM_HEIGHT))
#     frame = cv2.flip(frame, 1)
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(img_rgb)
#     current_time = time.time()
#     label = ""
#     confidence = 0
#     unknown_gesture = False

#     if results.multi_hand_landmarks:
#         last_hand_seen = current_time
#         # spoken_once is NO LONGER reset here. It should only be reset when new text is added or explicitly cleared.

#         for hand_landmarks in results.multi_hand_landmarks:
#             h, w, _ = frame.shape
#             x_list = [lm.x * w for lm in hand_landmarks.landmark]
#             y_list = [lm.y * h for lm in hand_landmarks.landmark]
#             padding = 80
#             x_min = max(0, int(min(x_list)) - padding)
#             x_max = min(w, int(max(x_list)) + padding)
#             y_min = max(0, int(min(y_list)) - padding)
#             y_max = min(h, int(max(y_list)) + padding)

#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             if flash_timer > 0:
#                 overlay = frame.copy()
#                 cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), -1)
#                 frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
#                 flash_timer -= 1
#             else:
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

#             landmark_input = extract_landmarks(hand_landmarks, (h, w))
#             if landmark_input.shape != (42,):
#                 continue
#             roi_input = np.expand_dims(landmark_input, axis=0)

#             if prediction_count == 0:
#                 prediction = model.predict(roi_input, verbose=0)[0]
#                 confidence = np.max(prediction)
#                 label = labels[np.argmax(prediction)]
#                 prediction_buffer.append(label)

#                 if confidence > 0.85:
#                     if label == prev_prediction:
#                         consistency_counter += 1
#                     else:
#                         consistency_counter = 1
#                         prev_prediction = label

#                     if consistency_counter >= required_consistency:
#                         if (current_time - recent_prediction_time > repeat_delay):
#                             # Only set spoken_once to False when a *new* word is added to typed_text
#                             # if typed_text.strip() == "" or label not in typed_text.strip().split():
#                             spoken_once = False     #
#                             typed_text += label + " "
#                             history.append(label)
#                             recent_prediction_time = current_time
#                             flash_timer = 5
#                             winsound.Beep(800, 100)
#                         consistency_counter = 0
#                     unknown_gesture = False
#                 else:
#                     prev_prediction = ""
#                     consistency_counter = 0
#                     unknown_gesture = True

#                 cv2.putText(frame, f"Prediction: {label} ({confidence:.2f})", (200, 40),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#     else:
#         cv2.putText(frame, "No hand detected", (10, 80),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     if unknown_gesture:
#         cv2.putText(frame, "❌ No known gesture found", (10, 50),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#     # Render overlay text
#     words = typed_text.strip().split()
#     max_width = 960
#     line = ""
#     y_pos = 360
#     for word in words:
#         test_line = line + word + " " if line else word + " " 
        
#         (text_width, text_height), _ = cv2.getTextSize(test_line.strip(), cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#         if text_width < max_width:
#             line = test_line
#         else:
#             cv2.putText(frame, line.strip(), (10, y_pos),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#             line = word + " "
#             y_pos += 35
#     if line:
#         cv2.putText(frame, line.strip(), (10, y_pos),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     else:
#         cv2.putText(frame, "[Waiting for prediction...]", (10, y_pos),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

#     # Speak text after idle ONLY IF spoken_once is False
#     if (current_time - last_hand_seen) > 5 and typed_text.strip() and not spoken_once:
#         speak_async(typed_text.strip())
#         spoken_once = True

#     prediction_count = (prediction_count + 1) % cooldown
#     return frame, typed_text.strip()

























#for using it in backend --- use this  (perfect working )

# import os
# import time
# import cv2
# import numpy as np
# import tensorflow as tf
# import pyttsx3
# import winsound
# import mediapipe as mp
# from collections import deque, Counter
# from tensorflow.keras.models import load_model
# from absl import logging

# # Suppress logs and warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# logging.set_verbosity(logging.FATAL)
# tf.get_logger().setLevel('ERROR')

# # Load model and labels
# model = load_model('model_landmark/sign_model_landmark.keras')
# with open('model_landmark/labels.txt', 'r') as f:
#     labels = [line.strip() for line in f.readlines()]

# # Webcam setup
# cap = cv2.VideoCapture(1)
# cap.set(3, 1280)
# cap.set(4, 560)

# # Mediapipe setup
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# # TTS setup
# engine = pyttsx3.init()
# engine.setProperty('rate', 120)
# engine.setProperty('volume', 1.0)
# engine.setProperty('voice', engine.getProperty('voices')[1].id)

# # Variables
# typed_text = ""
# history = []
# prediction_buffer = deque(maxlen=10)
# prev_prediction = ""
# consistency_counter = 0
# required_consistency = 3
# flash_timer = 0
# prediction_count = 0
# cooldown = 12
# recent_prediction_time = time.time()
# last_hand_seen = time.time()
# spoken_once = False
# repeat_delay = 2  # seconds before same word can be added again

# def extract_landmarks(landmarks, frame_shape):
#     h, w = frame_shape
#     return np.array([[lm.x, lm.y] for lm in landmarks.landmark]).flatten()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     frame = cv2.flip(frame, 1)
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(img_rgb)
#     current_time = time.time()
#     label = ""
#     confidence = 0
#     unknown_gesture = False

#     if results.multi_hand_landmarks:
#         last_hand_seen = current_time
#         spoken_once = False

#         for hand_landmarks in results.multi_hand_landmarks:
#             h, w, _ = frame.shape
#             x_list = [lm.x * w for lm in hand_landmarks.landmark]
#             y_list = [lm.y * h for lm in hand_landmarks.landmark]

#             # ROI logic from original predict.py
#             padding = 80
#             x_min = max(0, int(min(x_list)) - padding)
#             x_max = min(w, int(max(x_list)) + padding)
#             y_min = max(0, int(min(y_list)) - padding)
#             y_max = min(h, int(max(y_list)) + padding)

#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # Draw bounding box
#             if flash_timer > 0:
#                 overlay = frame.copy()
#                 cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), -1)
#                 frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
#                 flash_timer -= 1
#             else:
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

#             # Show ROI
#             roi_display = frame[y_min:y_max, x_min:x_max]
#             if roi_display.size > 0:
#                 roi_resized = cv2.resize(roi_display, (200, 200))
#                 cv2.imshow("ROI", roi_resized)

#             # Landmark input
#             landmark_input = extract_landmarks(hand_landmarks, (h, w))
#             if landmark_input.shape != (42,):
#                 continue
#             roi_input = np.expand_dims(landmark_input, axis=0)

#             if prediction_count == 0:
#                 prediction = model.predict(roi_input, verbose=0)[0]
#                 confidence = np.max(prediction)
#                 label = labels[np.argmax(prediction)]
#                 prediction_buffer.append(label)
#                 smoothed_label = Counter(prediction_buffer).most_common(1)[0][0]

#                 print(f"[PREDICTION] --> {label} ({confidence:.2f}) - Time: {time.strftime('%H:%M:%S')}")
#                 cv2.putText(frame, f"Prediction: {label} ({confidence:.2f})", (200, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#                 if confidence > 0.85:
#                     if label == prev_prediction:
#                         consistency_counter += 1
#                     else:
#                         consistency_counter = 1
#                         prev_prediction = label

#                     if consistency_counter >= required_consistency:
#                         if (current_time - recent_prediction_time > repeat_delay):
#                             typed_text += label + " "
#                             history.append(label)
#                             recent_prediction_time = current_time
#                             flash_timer = 5
#                             winsound.Beep(800, 100)
#                         consistency_counter = 0
#                     unknown_gesture = False
#                 else:
#                     prev_prediction = ""
#                     consistency_counter = 0
#                     unknown_gesture = True

#     else:
#         cv2.putText(frame, "No hand detected", (10, 90),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # Unknown gesture message
#     if unknown_gesture:
#         cv2.putText(frame, "❌ No known gesture found", (10, 60),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

#     # Typed text display
#     words = typed_text.strip().split()
#     max_width = 1000
#     line = ""
#     y_pos = 500
#     for word in words:
#         if cv2.getTextSize(line + word, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] < max_width:
#             line += word + " "
#         else:
#             cv2.putText(frame, line.strip(), (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#             line = word + " "
#             y_pos += 35
#     if line:
#         cv2.putText(frame, line.strip(), (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     else:
#         cv2.putText(frame, "[Waiting for prediction...]", (10, y_pos),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

#     # Speak text after idle
#     if (current_time - last_hand_seen) > 5 and typed_text.strip() and not spoken_once:
#         print(f"[TTS] Speaking: {typed_text.strip()}")
#         engine.say(typed_text.strip())
#         engine.runAndWait()
#         spoken_once = True

#     # Footer
#     cv2.putText(frame, "Press ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
#     cv2.putText(frame, "'R'", (75, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#     cv2.putText(frame, "=Reset", (105, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
#     cv2.putText(frame, "'Q'", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#     cv2.putText(frame, "=Quit", (230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
#     cv2.putText(frame, "'Backspace'", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
#     cv2.putText(frame, "=Delete", (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
#     cv2.putText(frame, f"DEBUG: typed_text='{typed_text}'", (10, 540),
#                 cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

#     cv2.imshow("Sign Language Translator", frame)
#     prediction_count = (prediction_count + 1) % cooldown

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('r'):
#         typed_text = ""
#         history = []
#         prev_prediction = ""
#         spoken_once = False
#     elif key == 8:  # Backspace
#         if history:
#             history.pop()
#             typed_text = " ".join(history) + " "

# cap.release()
# cv2.destroyAllWindows()
