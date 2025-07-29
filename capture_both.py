import os
import cv2
import numpy as np
import mediapipe as mp
import pyttsx3
import threading
import time


# from app import capturing_status
# import app
# capturing_status = app.capturing_status


# TTS Setup
tts = pyttsx3.init()
tts.setProperty('rate', 160)
tts.setProperty('voice', tts.getProperty('voices')[1].id)
tts_lock = threading.Lock()

capturing_done = False

def speak(text):
    def _speak():
        with tts_lock:
            try:
                tts.say(text)
                tts.runAndWait()
            except RuntimeError:
                pass
    threading.Thread(target=_speak).start() 
    

def capture_stream(word, capturing_status):

    global capturing_done
    capturing_done = False
    
    capturing_status[word] = "capturing"
    
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
    mp_drawing = mp.solutions.drawing_utils

    count = 0
    total = 200
    sample_interval = 0.1
    last_capture_time = time.time()

    lm_path = f"dataset_landmark/word_landmark/{word}"
    hy_path = f"dataset_hybrid/word_hybrid/{word}"
    lm_preview_path = os.path.join(lm_path, "preview")
    hy_preview_path = os.path.join(hy_path, "preview")

    os.makedirs(lm_path, exist_ok=True)
    os.makedirs(hy_path, exist_ok=True)
    os.makedirs(lm_preview_path, exist_ok=True)
    os.makedirs(hy_preview_path, exist_ok=True)
    
    
    # --- START OF UPDATED SECTION ---
    # Try camera 1 first
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("[CAMERA DEBUG] Camera 1 not found or busy. Trying Camera 0.")
        # Release if it was partially opened but failed
        if cap.isOpened():
            cap.release()
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW) # Try camera 0

    if cap.isOpened():
        print("[CAMERA DEBUG] Camera successfully opened.")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    else:
        print("[CRITICAL ERROR] No camera accessible. Capture will be interrupted.")
        speak("Camera not found")
        capturing_status[word] = "interrupted"
        return
    # --- END OF UPDATED SECTION ---


    # cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    # cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    # cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # # if not cap.isOpened():
    # #     cap = cv2.VideoCapture(0)
    
    # if not cap.isOpened():
    #     cap.release()  # Release failed capture before retrying
    #     # print("[CAMERA DEBUG] Camera 1 not found or busy. Trying Camera 0.")
    #     cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    #     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    #     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        
    # if not cap.isOpened():
    #     speak("Camera not found")
    #     capturing_status[word] = "interrupted"  # ✅ Mark as failed
    #     return
    
    

    # speak(f"Tracking hand for class {word}. Move your hand naturally.")

    prev_time = time.time()
    halfway_announced = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[FRAME DEBUG] Failed to read frame from camera. Skipping.")
            continue
        print("[FRAME DEBUG] Frame successfully read and being processed.")

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        current_time = time.time()
        fps = 1 / (current_time - prev_time)
        prev_time = current_time

        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            x_vals = [lm.x for lm in hand_landmarks.landmark]
            y_vals = [lm.y for lm in hand_landmarks.landmark]
            min_x, max_x = min(x_vals), max(x_vals)
            min_y, max_y = min(y_vals), max(y_vals)
            # padding = 0.05
            padding = 0.07 + np.random.uniform(0.01, 0.03)
            box_x1 = int(max((min_x - padding) * w, 0))
            box_y1 = int(max((min_y - padding) * h, 0))
            box_x2 = int(min((max_x + padding) * w, w))
            box_y2 = int(min((max_y + padding) * h, h))

            if time.time() - last_capture_time >= sample_interval:
                # Save landmarks
                landmark = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]
                np.save(os.path.join(lm_path, f"{count}.npy"), np.array(landmark))
                np.save(os.path.join(hy_path, f"{count}.npy"), np.array(landmark))

                # Save hybrid preview images
                roi = frame[box_y1:box_y2, box_x1:box_x2]
                roi = cv2.resize(roi, (200, 200))

                cv2.imwrite(os.path.join(lm_preview_path, f"{count}.jpg"), roi)
                cv2.imwrite(os.path.join(hy_preview_path, f"{count}.jpg"), roi)

                count += 1
                last_capture_time = time.time()

                # if not halfway_announced and count >= total // 2:
                #     halfway_announced = True
                #     speak("Halfway completed")
                
                if not halfway_announced and count >= total // 2:
                    halfway_announced = True
                    capturing_status[word] = "halfway"
                    speak("Halfway completed")
                    # time.sleep(0.5)  # allow TTS buffer
                    
        # Overlay info
        cv2.putText(frame, f"Class: {word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Samples: {count}/{total}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        # cv2.putText(frame, "Tracking hand... | Q to quit", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Draw bounding box
        if results.multi_hand_landmarks:
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)

        # cv2.imshow("Hand Capture", frame)
        
        if capturing_status.get(word) == "interrupted":
            print(f"[INFO] Capture for '{word}' interrupted by user via status flag.")
            break 

        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     capturing_status[word] = "interrupted"
        #     speak(f"Capture interrupted for {word} (via direct key press).")
        #     # time.sleep(0.5)
        #     break
        
        if count >= total:
            break
        
        # ✅ Yield encoded JPEG frame to frontend
        # cv2.putText(frame, "Test Stream", (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, buffer = cv2.imencode('.jpg', frame)
        # print("[DEBUG] Frame size:", len(buffer))
        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )

        
    cap.release()
    # cv2.destroyAllWindows()
    if capturing_status.get(word) != "interrupted": # If not already marked interrupted by user
        capturing_status[word] = "done" # Mark as done if capture completed
        speak(f"Capture session ended for {word}.")
    else:
        # If it was interrupted, the speak call is handled by the frontend
        pass 
    time.sleep(0.5)  # Let TTS finish before stream closes
    capturing_status[word] = "done"
    # speak(f"Capture session ended for {word}.")
    capturing_done = True













# 2nd change working only little changes needed ---
# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# import pyttsx3
# import threading
# import time

# # TTS Setup
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
#                 pass
#     threading.Thread(target=_speak).start()

# def capture_stream(word):
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.75, min_tracking_confidence=0.75)
#     mp_drawing = mp.solutions.drawing_utils

#     count = 0
#     total = 200
#     sample_interval = 0.1
#     last_capture_time = time.time()

#     # lm_path = f"dataset_landmark/word_landmark/{word}"
#     # hy_path = f"dataset_hybrid/word_hybrid/{word}"
#     # preview_path = os.path.join(hy_path, "preview")

#     # os.makedirs(lm_path, exist_ok=True)
#     # os.makedirs(hy_path, exist_ok=True)
#     # os.makedirs(preview_path, exist_ok=True)
    
#     lm_path = f"dataset_landmark/word_landmark/{word}"
#     hy_path = f"dataset_hybrid/word_hybrid/{word}"

#     # Create separate preview folders for landmark and hybrid
#     lm_preview_path = os.path.join(lm_path, "preview")
#     hy_preview_path = os.path.join(hy_path, "preview")

#     os.makedirs(lm_path, exist_ok=True)
#     os.makedirs(hy_path, exist_ok=True)
#     os.makedirs(lm_preview_path, exist_ok=True)
#     os.makedirs(hy_preview_path, exist_ok=True)

#     cap = cv2.VideoCapture(1)
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#     if not cap.isOpened():
#         cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         speak("Camera not found")
#         return

#     speak(f"Tracking hand for class {word}. Move your hand naturally.")

#     prev_time = time.time()
#     halfway_announced = False

#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         frame = cv2.flip(frame, 1)
#         h, w, _ = frame.shape
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(rgb)

#         current_time = time.time()
#         fps = 1 / (current_time - prev_time)
#         prev_time = current_time

#         if results.multi_hand_landmarks:
#             hand_landmarks = results.multi_hand_landmarks[0]
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             x_vals = [lm.x for lm in hand_landmarks.landmark]
#             y_vals = [lm.y for lm in hand_landmarks.landmark]
#             min_x, max_x = min(x_vals), max(x_vals)
#             min_y, max_y = min(y_vals), max(y_vals)
#             padding = 0.05
#             box_x1 = int(max((min_x - padding) * w, 0))
#             box_y1 = int(max((min_y - padding) * h, 0))
#             box_x2 = int(min((max_x + padding) * w, w))
#             box_y2 = int(min((max_y + padding) * h, h))

#             if time.time() - last_capture_time >= sample_interval:
#                 landmark = [coord for lm in hand_landmarks.landmark for coord in (lm.x, lm.y)]
#                 np.save(os.path.join(lm_path, f"{count}.npy"), np.array(landmark))
#                 np.save(os.path.join(hy_path, f"{count}.npy"), np.array(landmark))

#                 roi = frame[box_y1:box_y2, box_x1:box_x2]
#                 roi = cv2.resize(roi, (200, 200))
#                 # cv2.imwrite(os.path.join(preview_path, f"{count}.jpg"), roi)
                
#                 # Save preview images in both landmark and hybrid preview folders
#                 cv2.imwrite(os.path.join(lm_preview_path, f"{count}.jpg"), roi)
#                 cv2.imwrite(os.path.join(hy_preview_path, f"{count}.jpg"), roi)


#                 count += 1
#                 last_capture_time = time.time()

#                 if not halfway_announced and count >= total // 2:
#                     halfway_announced = True
#                     speak("Halfway completed")

#         cv2.putText(frame, f"Class: {word}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#         cv2.putText(frame, f"Samples: {count}/{total}", (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 0), 2)
#         cv2.putText(frame, f"FPS: {fps:.2f}", (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
#         cv2.putText(frame, "Tracking hand... | Q to quit", (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

#         cv2.imshow("Hand Capture", frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
#         if count >= total:
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     speak(f"Capture session ended for {word}.")













#1st change ------
# import os
# import cv2
# import numpy as np
# import mediapipe as mp
# import pyttsx3
# import threading

# # TTS Setup
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
#                 pass
#     threading.Thread(target=_speak).start()

# def capture_stream(word):
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(max_num_hands=1)
#     count = 0
#     total = 200

#     lm_path = f"dataset_landmark/word_landmark/{word}"
#     hy_path = f"dataset_hybrid/word_hybrid/{word}"
    
#     preview_path = os.path.join(hy_path, "preview")

#     os.makedirs(lm_path, exist_ok=True)
#     os.makedirs(hy_path, exist_ok=True)
#     os.makedirs(preview_path, exist_ok=True)

#     cap = cv2.VideoCapture(1)
#     if not cap.isOpened():
#         cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         speak("Camera not found")
#         return

#     speak(f"Capturing gesture for {word}")

#     while count < total:
#         ret, frame = cap.read()
#         if not ret:
#             continue

#         frame = cv2.flip(frame, 1)
#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = hands.process(rgb)

#         if results.multi_hand_landmarks:
#             for hand in results.multi_hand_landmarks:
#                 h, w, _ = frame.shape
#                 landmark = np.array([[lm.x, lm.y] for lm in hand.landmark]).flatten()

#                 # Save landmark
#                 np.save(os.path.join(lm_path, f"{count:03}.npy"), landmark)

#                 # Save hybrid
#                 np.save(os.path.join(hy_path, f"{count:03}.npy"), landmark)
#                 x = [lm.x * w for lm in hand.landmark]
#                 y = [lm.y * h for lm in hand.landmark]
#                 x_min, x_max = int(min(x)) - 20, int(max(x)) + 20
#                 y_min, y_max = int(min(y)) - 20, int(max(y)) + 20
#                 x_min, x_max = max(0, x_min), min(w, x_max)
#                 y_min, y_max = max(0, y_min), min(h, y_max)

#                 roi = frame[y_min:y_max, x_min:x_max]
#                 roi = cv2.resize(roi, (200, 200))
#                 cv2.imwrite(os.path.join(preview_path, f"{count:03}.jpg"), roi)

#                 count += 1

#         cv2.putText(frame, f"Capturing {count}/{total}", (10, 30),
#                     cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
#         _, buffer = cv2.imencode('.jpg', frame)
#         yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")

#     cap.release()
#     speak(f"Capture complete for {word}")
