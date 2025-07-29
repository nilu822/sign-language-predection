# this code is just for testing with more features included --- original is given below
import os
import time
import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import winsound
import mediapipe as mp
from collections import deque, Counter
from tensorflow.keras.models import load_model
from absl import logging

# Suppress logs and warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
logging.set_verbosity(logging.FATAL)
tf.get_logger().setLevel('ERROR')

# Load model and labels
model = load_model('model/model.h5')
with open('model/labels.txt', 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Webcam setup
cap = cv2.VideoCapture(1)
cap.set(3, 1020)
cap.set(4, 400)

# Mediapipe Hands
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# Text-to-speech
engine = pyttsx3.init()
engine.setProperty('rate', 120)
engine.setProperty('volume', 1.0)
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[1].id)

# Variables
typed_text = ""
history = []
prediction_buffer = deque(maxlen=10)
prev_prediction = ""
consistency_counter = 0
required_consistency = 3
flash_timer = 0
prediction_count = 0
cooldown = 12
last_typed = ""
last_hand_seen = time.time()
spoken_once = False
recent_prediction_time = time.time()

def preprocess_hand(roi):
    roi = cv2.resize(roi, (96, 96))
    roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    roi = roi / 255.0
    return np.expand_dims(roi, axis=0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    current_time = time.time()
    label = ""
    confidence = 0

    if results.multi_hand_landmarks:
        last_hand_seen = current_time
        spoken_once = False

        for hand_landmarks in results.multi_hand_landmarks:
            h, w, _ = frame.shape
            x_list = [lm.x * w for lm in hand_landmarks.landmark]
            y_list = [lm.y * h for lm in hand_landmarks.landmark]

            padding = 80
            x_min = max(0, int(min(x_list)) - padding)
            x_max = min(w, int(max(x_list)) + padding)
            y_min = max(0, int(min(y_list)) - padding)
            y_max = min(h, int(max(y_list)) + padding)

            roi_color = frame[y_min:y_max, x_min:x_max]

            # Lighting normalization
            yuv = cv2.cvtColor(roi_color, cv2.COLOR_BGR2YUV)
            yuv[:, :, 0] = cv2.equalizeHist(yuv[:, :, 0])
            roi_color = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

            roi_input = preprocess_hand(roi_color)

            if roi_input.shape[1] == 0 or roi_input.shape[2] == 0:
                continue

            if flash_timer > 0:
                overlay = frame.copy()
                cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), -1)
                frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
                flash_timer -= 1
            else:
                cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 4)

            cv2.imshow("ROI", roi_color)

            if prediction_count == 0:
                prediction = model.predict(roi_input, verbose=0)[0]
                confidence = np.max(prediction)
                label = labels[np.argmax(prediction)]
                prediction_buffer.append(label)
                smoothed_label = Counter(prediction_buffer).most_common(1)[0][0]

                print(f"[PREDICTION] {label} ({confidence:.2f}) - Time: {time.strftime('%H:%M:%S')}")

                cv2.putText(frame, f"Prediction: {label} ({confidence:.2f})", (200, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if (current_time - recent_prediction_time) < 1.5:
                    break

                if confidence > 0.85:
                    if label == prev_prediction:
                        consistency_counter += 1
                    else:
                        consistency_counter = 1
                        prev_prediction = label

                    if consistency_counter >= required_consistency:
                        if label != last_typed and (current_time - recent_prediction_time > 1.5):
                            if label.lower() == 'space':
                                typed_text += " "
                                history.append(' ')
                            else:
                                typed_text += label + " "
                                history.append(label)
                            last_typed = label
                            recent_prediction_time = current_time
                            flash_timer = 5
                            winsound.Beep(800, 100)
                        consistency_counter = 0
                else:
                    prev_prediction = ""
                    consistency_counter = 0
                    cv2.putText(frame, "Low confidence", (10, 90),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
    else:
        cv2.putText(frame, "No hand detected", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Display typed text with wrapping
    words = typed_text.strip().split()
    max_width = 580
    line = ""
    y_pos = 420
    for word in words:
        if cv2.getTextSize(line + word, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] < max_width:
            line += word + " "
        else:
            cv2.putText(frame, line.strip(), (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            line = word + " "
            y_pos += 35
    if line:
        cv2.putText(frame, line.strip(), (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    else:
        cv2.putText(frame, "[Waiting for prediction...]", (10, y_pos),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

    if (current_time - last_hand_seen) > 5 and typed_text.strip() and not spoken_once:
        print(f"[TTS] Speaking: {typed_text.strip()}")
        engine.say(typed_text.strip())
        engine.runAndWait()
        spoken_once = True

    # Footer
    cv2.putText(frame, "Press ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, "'R'", (75, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, "=Reset", (105, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, "'Q'", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "=Quit", (230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, "'Backspace'", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
    cv2.putText(frame, "=Delete", (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
    cv2.putText(frame, f"DEBUG: typed_text='{typed_text}'", (10, 460),
                cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    cv2.imshow("Sign Language Translator", frame)
    prediction_count = (prediction_count + 1) % cooldown

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('r'):
        typed_text = ""
        history = []
        last_typed = ""
        spoken_once = False
    elif key == 8:  # Backspace
        if history:
            last = history.pop()
            if last in labels:
                typed_text = " ".join(typed_text.strip().split(" ")[:-1]) + " "
            else:
                typed_text = typed_text[:-2]

cap.release()
cv2.destroyAllWindows()

























# -- this is the original working code --- was using this -- remember
# import os
# import time
# import cv2
# import numpy as np
# import tensorflow as tf
# import pyttsx3
# import winsound
# import mediapipe as mp
# from collections import deque, Counter  # ADD THIS FOR SMOOTHING
# from tensorflow.keras.models import load_model
# from absl import logging

# # Suppress logs and warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# logging.set_verbosity(logging.FATAL)
# tf.get_logger().setLevel('ERROR')

# # Load model and labels
# model = load_model('model/model.h5')
# with open('model/labels.txt', 'r') as f:
#     labels = [line.strip() for line in f.readlines()]

# # Webcam setup
# cap = cv2.VideoCapture(1)
# # cap.set(3, 640)       # 1st setting of camera resoltuion
# # cap.set(4, 480)
    
# # cap.set(3, 1280)      #2nd  setting of camera resoltuion
# # cap.set(4, 720)

# cap.set(3, 1020)
# cap.set(4, 400)

# # Mediapipe Hands
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# # Text-to-speech
# engine = pyttsx3.init()
# engine.setProperty('rate', 120)
# engine.setProperty('volume', 1.0)
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)

# # Variables
# typed_text = ""
# history = []
# # Smooth prediction with buffer
# prediction_buffer = deque(maxlen=10)
# prev_prediction = ""
# consistency_counter = 0
# required_consistency = 3
# flash_timer = 0
# prediction_count = 0
# prediction_delay = 2.0
# # recent_prediction_time=0
# cooldown = 12
# last_typed = ""
# last_hand_seen = time.time()
# spoken_once = False
# recent_prediction_time = time.time()

# def apply_skin_mask(frame):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_skin = np.array([0, 20, 70], dtype=np.uint8)
#     upper_skin = np.array([20, 255, 255], dtype=np.uint8)
#     mask = cv2.inRange(hsv, lower_skin, upper_skin)
#     result = cv2.bitwise_and(frame, frame, mask=mask)
#     return result

# def preprocess_hand(roi):
#     roi = cv2.resize(roi, (96, 96))
#     roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#     roi = roi / 255.0
#     return np.expand_dims(roi, axis=0)

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

#     if results.multi_hand_landmarks:
#         last_hand_seen = current_time
#         spoken_once = False

#         for hand_landmarks in results.multi_hand_landmarks:
#             h, w, _ = frame.shape
#             x_list = [lm.x * w for lm in hand_landmarks.landmark]
#             y_list = [lm.y * h for lm in hand_landmarks.landmark]

#             padding = 80        #50
#             x_min = max(0, int(min(x_list)) - padding)
#             x_max = min(w, int(max(x_list)) + padding)
#             y_min = max(0, int(min(y_list)) - padding)
#             y_max = min(h, int(max(y_list)) + padding)

#             roi_color = frame[y_min:y_max, x_min:x_max]
#             roi_input = preprocess_hand(roi_color)

#             if roi_input is None or roi_input.shape[1] == 0 or roi_input.shape[2] == 0:
#                 continue

#             if flash_timer > 0:
#                 overlay = frame.copy()
#                 cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), -1)
#                 frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
#                 flash_timer -= 1
#             else:
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 4)

#             # print(f"ROI shape: {roi_color.shape}")        #just to showing the ROI shape 
#             cv2.imshow("ROI", roi_color)

#             if prediction_count == 0:
#                 prediction = model.predict(roi_input, verbose=0)[0]     # [0] is added
#                 # added this --
#                 top3 = np.argsort(prediction)[-3:][::-1]
                
                
#                 # print("[TOP 3 CLASSES]")
#                 # for i in top3:
#                 #     print(f"{labels[i]}: {prediction[i]:.4f}")
                
                
#                 confidence = np.max(prediction)
                
#                 #ADDED ---
#                 label = labels[np.argmax(prediction)]
#                 prediction_buffer.append(label)
#                 smoothed_label = Counter(prediction_buffer).most_common(1)[0][0]

#                 # label = labels[np.argmax(prediction)]
#                 # print(f"[DEBUG] Predicted: {label}, Confidence: {confidence:.2f}")
#                 print(f"[PREDICTION] {label} ({confidence:.2f}) - Time: {time.strftime('%H:%M:%S')}")


#                 # cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 60),
#                 #             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
#                 cv2.putText(frame, f"Prediction:{label} ({confidence:.2f})", (200, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                
#                 # Prevent frequent predictions within 1.5 seconds
#                 if (current_time - recent_prediction_time) < 1.5:
#                     break

#                 if confidence > 0.85:
#                     if label == prev_prediction:
#                         consistency_counter += 1
#                     else:
#                         consistency_counter = 1
#                         prev_prediction = label
                        
#                     if consistency_counter >= required_consistency:
#                         if label != last_typed and (current_time - recent_prediction_time > 1.5):
#                         # if label != last_typed or (current_time - recent_prediction_time > 1.5):
#                             if label.lower() == 'space':
#                                 typed_text += " "
#                                 history.append(' ') 
#                             else:
#                                 # Avoid repeating last word
#                                 # if not typed_text.endswith(f"{label} "):
#                                 typed_text += label + " "
#                                 history.append(label)
#                             last_typed = label
#                             recent_prediction_time = current_time
#                             flash_timer = 5
#                             winsound.Beep(800, 100)  # Beep sound
#                         consistency_counter = 0

#                     # if consistency_counter >= required_consistency:
#                     #     if label.lower() == 'space':
#                     #         typed_text += " "
#                     #         history.append(" ")
#                     #     else:
#                     #         typed_text += label
#                     #         history.append(label)

#                     #     last_typed = label
#                     #     flash_timer = 5
#                     #     winsound.Beep(800, 100)
#                     #     consistency_counter = 0
#                 else:
#                     prev_prediction = ""
#                     consistency_counter = 0
#                     cv2.putText(frame, "Low confidence", (10, 90),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#     else:
#         cv2.putText(frame, "No hand detected", (10, 90),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # Typed text display with wrapping
    
#     words = typed_text.strip().split()
#     max_width = 580
#     line = ""
#     y_pos = 420
#     for word in words:
#         if cv2.getTextSize(line + word, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] < max_width:
#             line += word + " "
#         else:
#             cv2.putText(frame, line.strip(), (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#             line = word + " "
#             y_pos += 35
#     if line:
#         cv2.putText(frame, line.strip(), (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

    
    
#     # max_width = 580
#     # y_pos = 420
#     # if typed_text:
#     #     line = ""
#     #     for char in typed_text:
#     #         test_line = line + char
#     #         (text_width, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#     #         if text_width < max_width:
#     #             line = test_line
#     #         else:
#     #             cv2.putText(frame, line, (10, y_pos),
#     #                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     #             y_pos += 35
#     #             line = char
#     #     if line:
#     #         cv2.putText(frame, line, (10, y_pos),
#     #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    
    
#     else:
#         cv2.putText(frame, "[Waiting for prediction...]", (10, y_pos),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

#     # TTS after delay
#     if (current_time - last_hand_seen) > 5 and typed_text.strip() and not spoken_once:
#         print(f"[TTS] Speaking: {typed_text.strip()}")
#         engine.say(typed_text.strip())
#         engine.runAndWait()
#         spoken_once = True

#     # Footer instructions
#     cv2.putText(frame, "Press ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
#     cv2.putText(frame, "'R'", (75, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#     cv2.putText(frame, "=Reset", (105, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
#     cv2.putText(frame, "'Q'", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#     cv2.putText(frame, "=Quit", (230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
#     cv2.putText(frame, "'Backspace'", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
#     cv2.putText(frame, "=Delete", (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

#     cv2.putText(frame, f"DEBUG: typed_text='{typed_text}'", (10, 460),
#                 cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

#     cv2.imshow("Sign Language Translator", frame)
#     prediction_count = (prediction_count + 1) % cooldown

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('r'):
#         typed_text = ""
#         history = []
#         last_typed = ""
#         spoken_once = False
#     elif key == 8:  # Backspace
#         if history:
#             last = history.pop()
#             if last in labels:
#                 typed_text = " ".join(typed_text.strip().split(" ")[:-1]) + " "
#             else:
#                 typed_text = typed_text[:-2]

# cap.release()
# cv2.destroyAllWindows()











































# working only space problem screen is coming -----
# import os
# import time
# import cv2
# import numpy as np
# import tensorflow as tf
# import pyttsx3
# import winsound
# import mediapipe as mp
# from tensorflow.keras.models import load_model
# from absl import logging

# # Suppress logs and warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# logging.set_verbosity(logging.FATAL)
# tf.get_logger().setLevel('ERROR')

# # Load model and labels
# model = load_model('model/model.h5')
# with open('model/labels.txt', 'r') as f:
#     labels = [line.strip() for line in f.readlines()]

# # Webcam setup
# cap = cv2.VideoCapture(1)
# cap.set(3, 640)
# cap.set(4, 480)

# # Mediapipe Hands
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# # Text-to-speech
# engine = pyttsx3.init()
# engine.setProperty('rate', 120)
# engine.setProperty('volume', 1.0)
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)

# # Variables
# typed_text = ""
# history = []
# prev_prediction = ""
# consistency_counter = 0
# required_consistency = 3
# flash_timer = 0
# prediction_count = 0
# cooldown = 8
# last_typed = ""
# last_hand_seen = time.time()
# spoken_once = False

# def apply_skin_mask(frame):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_skin = np.array([0, 20, 70], dtype=np.uint8)
#     upper_skin = np.array([20, 255, 255], dtype=np.uint8)
#     mask = cv2.inRange(hsv, lower_skin, upper_skin)
#     result = cv2.bitwise_and(frame, frame, mask=mask)
#     return result

# def preprocess_hand(roi):
#     roi = cv2.resize(roi, (96, 96))
#     roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#     roi = roi / 255.0
#     return np.expand_dims(roi, axis=0)

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

#     if results.multi_hand_landmarks:
#         last_hand_seen = current_time
#         spoken_once = False

#         for hand_landmarks in results.multi_hand_landmarks:
#             h, w, _ = frame.shape
#             x_list = [lm.x * w for lm in hand_landmarks.landmark]
#             y_list = [lm.y * h for lm in hand_landmarks.landmark]

#             padding = 50
#             x_min = max(0, int(min(x_list)) - padding)
#             x_max = min(w, int(max(x_list)) + padding)
#             y_min = max(0, int(min(y_list)) - padding)
#             y_max = min(h, int(max(y_list)) + padding)

#             roi_color = frame[y_min:y_max, x_min:x_max]
            
#             # roi_masked = apply_skin_mask(roi_color)
#             # roi_input = preprocess_hand(roi_masked)
            
#             roi_input = preprocess_hand(roi_color)

#             if roi_input is None or roi_input.shape[1] == 0 or roi_input.shape[2] == 0:
#                 continue

#             if flash_timer > 0:
#                 overlay = frame.copy()
#                 cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), -1)
#                 frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
#                 flash_timer -= 1
#             else:
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 4)

#             cv2.imshow("ROI", roi_color)

#             if prediction_count == 0:
#                 prediction = model.predict(roi_input, verbose=0)
#                 confidence = np.max(prediction)
#                 label = labels[np.argmax(prediction)]
#                 print(f"[DEBUG] Predicted: {label}, Confidence: {confidence:.2f}")

#                 # Always show label and confidence
#                 cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 60),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#                 if confidence > 0.5:        # 0.8
#                     if label == prev_prediction:
#                         consistency_counter += 1
#                     else:
#                         consistency_counter = 1
#                         prev_prediction = label

#                     if consistency_counter >= required_consistency:
#                         if label.lower() == 'space':
#                             typed_text += " "
#                             history.append(" ")
#                         else:
#                             typed_text += label
#                             history.append(label)

#                         last_typed = label
#                         flash_timer = 5
#                         winsound.Beep(800, 100)
#                         consistency_counter = 0
#                 else:
#                     prev_prediction = ""
#                     consistency_counter = 0
#                     cv2.putText(frame, "Low confidence", (10, 90),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
#     else:
#         cv2.putText(frame, "No hand detected", (10, 90),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # Typed text display with wrapping
#     max_width = 580
#     y_pos = 420
#     # if typed_text.strip():
#     if typed_text:
#         # words = typed_text.strip().split()
#         words = typed_text.split(' ')
#         # line = ""
#         for word in words:
#             # test_line = line + word + " "
#             test_line = (line + word + " ").strip()
#             (text_width, _), _ = cv2.getTextSize(test_line, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#             if text_width < max_width:
#                 # line = test_line
#                 line += word + " "
#             else:
#                 cv2.putText(frame, line.strip(), (10, y_pos),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#                 y_pos += 35
#                 line = word + " "
#         if line:
#             cv2.putText(frame, line.strip(), (10, y_pos),
#                         cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#     else:
#         cv2.putText(frame, "[Waiting for prediction...]", (10, y_pos),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.8, (100, 100, 100), 2)

#     # TTS after delay
#     if (current_time - last_hand_seen) > 5 and typed_text.strip() and not spoken_once:
#         print(f"[TTS] Speaking: {typed_text.strip()}")
#         engine.say(typed_text.strip())
#         engine.runAndWait()
#         spoken_once = True

#     # Footer instructions
#     cv2.putText(frame, "Press ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
#     cv2.putText(frame, "'R'", (75, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#     cv2.putText(frame, "=Reset", (105, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
#     cv2.putText(frame, "'Q'", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#     cv2.putText(frame, "=Quit", (230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
#     cv2.putText(frame, "'Backspace'", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
#     cv2.putText(frame, "=Delete", (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

#     # Debug display
#     cv2.putText(frame, f"DEBUG: typed_text='{typed_text}'", (10, 460),
#                 cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

#     cv2.imshow("Sign Language Translator", frame)
#     prediction_count = (prediction_count + 1) % cooldown

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('r'):
#         typed_text = ""
#         history = []
#         last_typed = ""
#         spoken_once = False
#     elif key == 8:  # Backspace
#         if history:
#             last = history.pop()
#             if last == ' ':
#                 typed_text = typed_text.rstrip()[:-1] + " "
#             else:
#                 typed_text = typed_text[:-len(last)]

# cap.release()
# cv2.destroyAllWindows()























# ✅ Updated predict.py with prediction label and accuracy at the top - 2nd changes
# import os
# import time
# import cv2
# import numpy as np
# import tensorflow as tf
# import pyttsx3
# import winsound
# import mediapipe as mp
# from tensorflow.keras.models import load_model
# from absl import logging

# # Suppress all logs and warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# logging.set_verbosity(logging.FATAL)
# tf.get_logger().setLevel('ERROR')

# # Load model and labels
# model = load_model('model/model.h5')
# with open('model/labels.txt', 'r') as f:
#     labels = [line.strip() for line in f.readlines()]

# # Camera
# cap = cv2.VideoCapture(1)
# cap.set(3, 640)
# cap.set(4, 480)

# # Mediapipe
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# # Text-to-Speech
# engine = pyttsx3.init()
# engine.setProperty('rate', 120)
# engine.setProperty('volume', 1.0)
# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)

# # Variables
# typed_text = ""
# history = []
# prev_prediction = ""
# consistency_counter = 0
# required_consistency = 3
# last_typed = ""
# last_hand_seen = time.time()
# spoken_once = False
# flash_timer = 0
# prediction_count = 0
# cooldown = 10

# def apply_skin_mask(frame):
#     hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
#     lower_skin = np.array([0, 20, 70], dtype=np.uint8)
#     upper_skin = np.array([20, 255, 255], dtype=np.uint8)
#     mask = cv2.inRange(hsv, lower_skin, upper_skin)
#     result = cv2.bitwise_and(frame, frame, mask=mask)
#     return result

# def preprocess_hand(roi):
#     roi = cv2.resize(roi, (96, 96))
#     roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#     roi = roi / 255.0
#     return np.expand_dims(roi, axis=0)

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

#     if results.multi_hand_landmarks:
#         last_hand_seen = current_time
#         spoken_once = False

#         for hand_landmarks in results.multi_hand_landmarks:
#             h, w, _ = frame.shape
#             x_list = [lm.x * w for lm in hand_landmarks.landmark]
#             y_list = [lm.y * h for lm in hand_landmarks.landmark]

#             padding = 50
#             x_min = max(0, int(min(x_list)) - padding)
#             x_max = min(w, int(max(x_list)) + padding)
#             y_min = max(0, int(min(y_list)) - padding)
#             y_max = min(h, int(max(y_list)) + padding)

#             roi_color = frame[y_min:y_max, x_min:x_max]
#             roi_masked = apply_skin_mask(roi_color)
#             roi_input = preprocess_hand(roi_masked)

#             if roi_input is None or roi_input.shape[1] == 0 or roi_input.shape[2] == 0:
#                 continue

#             # Draw ROI rectangle
#             if flash_timer > 0:
#                 overlay = frame.copy()
#                 cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), -1)
#                 frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
#                 flash_timer -= 1
#             else:
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 4)

#             cv2.imshow("ROI", roi_color)

#             if prediction_count == 0:
#                 prediction = model.predict(roi_input, verbose=0)
#                 confidence = np.max(prediction)
#                 label = labels[np.argmax(prediction)]
#                 print(f"[DEBUG] Predicted: {label}, Confidence: {confidence:.2f}")

#                 if confidence > 0.85:
#                     if label == prev_prediction:
#                         consistency_counter += 1
#                     else:
#                         consistency_counter = 1
#                         prev_prediction = label

#                     if consistency_counter >= required_consistency:
#                         if label != last_typed:
#                             if label.lower() == 'space':
#                                 typed_text += " "
#                             else:
#                                 typed_text += label + " "
#                             history.append(label)
#                             last_typed = label
#                             flash_timer = 5
#                             winsound.Beep(800, 150)
#                             print("[DEBUG] Prediction confirmed and beep played")
#                         consistency_counter = 0
#                 else:
#                     prev_prediction = ""
#                     consistency_counter = 0
#                     cv2.putText(frame, "Low confidence", (10, 90),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#                 cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 60),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     else:
#         cv2.putText(frame, "No hand detected", (10, 90),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # Display typed text with wrapping
#     words = typed_text.strip().split()
#     max_width = 580
#     line = ""
#     y_pos = 420
#     for word in words:
#         size = cv2.getTextSize(line + word, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0]
#         if size < max_width:
#             line += word + " "
#         else:
#             cv2.putText(frame, line.strip(), (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#             line = word + " "
#             y_pos += 35
#     if line:
#         cv2.putText(frame, line.strip(), (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#     # Speak after delay
#     if (current_time - last_hand_seen) > 5 and typed_text.strip() and not spoken_once:
#         print("[DEBUG] Speaking typed text")
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

#     cv2.imshow("Sign Language Translator", frame)
#     prediction_count = (prediction_count + 1) % cooldown

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('r'):
#         typed_text = ""
#         history = []
#         last_typed = ""
#         spoken_once = False
#     elif key == 8:  # Backspace
#         if history:
#             last = history.pop()
#             if last in labels:
#                 typed_text = " ".join(typed_text.strip().split(" ")[:-1]) + " "
#             else:
#                 typed_text = typed_text[:-2]

# cap.release()
# cv2.destroyAllWindows()


























# working code with sound - very imp - i was this earlier ( correct code ) -1
# import os
# import time
# import cv2
# import numpy as np
# import tensorflow as tf
# import absl.logging
# import pyttsx3
# import winsound
# from tensorflow.keras.models import load_model
# import mediapipe as mp

# # Suppress all logs and warnings
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# absl.logging.set_verbosity(absl.logging.FATAL)
# tf.get_logger().setLevel('ERROR')

# # Load model and labels
# model = load_model('model/model.h5')
# with open('model/labels.txt', 'r') as f:
#     labels = [line.strip() for line in f.readlines()]

# # Camera
# cap = cv2.VideoCapture(1)
# cap.set(3, 640)
# cap.set(4, 480)

# # Mediapipe
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)

# # Variables
# typed_text = ""
# history = []
# prev_prediction = ""
# prediction_count = 0
# cooldown = 10
# flash_timer = 0
# consistency_counter = 0
# required_consistency = 3
# recent_prediction_time = 0
# prediction_delay = 2.0
# last_typed = ""
# last_hand_seen = time.time()
# spoken_once = False

# # Text-to-speech
# engine = pyttsx3.init()
# engine.setProperty('rate', 120)     # Optional: adjust speech speed
# engine.setProperty('volume', 1.0)   # 🔊 Set max volume

# voices = engine.getProperty('voices')
# engine.setProperty('voice', voices[1].id)  # 👈 Pick index from printed list



# def preprocess_hand(frame, x_min, y_min, x_max, y_max):
#     hand_roi = frame[y_min:y_max, x_min:x_max]
#     if hand_roi.size == 0:
#         return None
#     roi = cv2.resize(hand_roi, (64, 64))
#     roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
#     roi = roi / 255.0
#     return np.expand_dims(roi, axis=0), hand_roi

# while True:
#     ret, frame = cap.read()
#     frame = cv2.flip(frame, 1)
#     img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(img_rgb)
#     current_time = time.time()
#     label = ""
#     confidence = 0

#     if results.multi_hand_landmarks:
#         last_hand_seen = current_time
#         spoken_once = False

#         for hand_landmarks in results.multi_hand_landmarks:
#             h, w, _ = frame.shape
#             x_list = [lm.x * w for lm in hand_landmarks.landmark]
#             y_list = [lm.y * h for lm in hand_landmarks.landmark]

#             padding = 50  # Increased padding for full hand
#             x_min = max(0, int(min(x_list)) - padding)
#             x_max = min(w, int(max(x_list)) + padding)
#             y_min = max(0, int(min(y_list)) - padding)
#             y_max = min(h, int(max(y_list)) + padding)

#             roi_input, roi_preview = preprocess_hand(frame, x_min, y_min, x_max, y_max)
#             if roi_input is None:
#                 continue

#             if flash_timer > 0:
#                 overlay = frame.copy()
#                 cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (0, 255, 255), -1)
#                 frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
#                 flash_timer -= 1
#             else:
#                 cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 4)

#             cv2.imshow("ROI", roi_preview)

#             if prediction_count == 0:
#                 prediction = model.predict(roi_input, verbose=0)
#                 confidence = np.max(prediction)
#                 label = labels[np.argmax(prediction)]

#                 print(f"[DEBUG] Predicted: {label}, Confidence: {confidence:.2f}")
#                 if confidence > 0.85:
#                     if label == prev_prediction:
#                         consistency_counter += 1
#                     else:
#                         consistency_counter = 1
#                         prev_prediction = label

                    # if consistency_counter >= required_consistency:
                    #     if label != last_typed or (current_time - recent_prediction_time > prediction_delay):
                    #         if label.lower() == 'space':
                    #             typed_text += " "
                    #         else:
                    #             typed_text += label + " "
                    #         history.append(label)
                    #         last_typed = label
                    #         recent_prediction_time = current_time
                    #         flash_timer = 5
                    #         winsound.Beep(800, 100)  # Beep sound
                    #     consistency_counter = 0
#                 else:
#                     prev_prediction = ""
#                     consistency_counter = 0
#                     cv2.putText(frame, "No confident prediction", (10, 90),
#                                 cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#                 cv2.putText(frame, f"{label} ({confidence:.2f})", (10, 60),
#                             cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

#             mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#     else:
#         cv2.putText(frame, "No hand detected", (10, 90),
#                     cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

#     # Display typed text wrapped by screen width
    # words = typed_text.strip().split()
    # max_width = 580
    # line = ""
    # y_pos = 420
    # for word in words:
    #     if cv2.getTextSize(line + word, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0][0] < max_width:
    #         line += word + " "
    #     else:
    #         cv2.putText(frame, line.strip(), (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
    #         line = word + " "
    #         y_pos += 35
    # if line:
    #     cv2.putText(frame, line.strip(), (10, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

#     # Speak after delay with no hand
#     if (current_time - last_hand_seen) > 5 and typed_text.strip() and not spoken_once:
#         engine.say(typed_text.strip())
#         engine.runAndWait()
#         spoken_once = True

#     # Footer
#     cv2.putText(frame, "Press ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
#     cv2.putText(frame, "'R'", (75, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
#     cv2.putText(frame, "=Reset  ", (105, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
#     cv2.putText(frame, "'Q'", (200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
#     cv2.putText(frame, "=Quit  ", (230, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)
#     cv2.putText(frame, "'Backspace'", (320, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
#     cv2.putText(frame, "=Delete", (440, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

#     cv2.imshow("Sign Language Translator", frame)
#     prediction_count = (prediction_count + 1) % cooldown

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         break
#     elif key == ord('r'):
#         typed_text = ""
#         history = []
#         last_typed = ""
#         spoken_once = False
    # elif key == 8:  # Backspace
    #     if history:
    #         last = history.pop()
    #         if last in labels:
    #             typed_text = " ".join(typed_text.strip().split(" ")[:-1]) + " "
    #         else:
    #             typed_text = typed_text[:-2]

# cap.release()
# cv2.destroyAllWindows()

