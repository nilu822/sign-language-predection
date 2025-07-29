from flask import Flask, render_template, request, Response, send_file, redirect, url_for, stream_with_context
import os, cv2, io, zipfile, subprocess, time, shutil


# ✅ Step 1: Define this first to avoid circular import
capturing_status = {}

# ✅ Step 2: Then import capture_both after defining capturing_status
import capture_both
from capture_both import capture_stream as capture_generator

# Import capture
# from capture_hybrid_data import capture_gesture_data as capture_hybrid
# from capture_landmark_data import capture_gesture_data as capture_landmark
# from capture_both import capture_stream
# from capture_both import capture_stream as capture_generator

# import capture_both



# Import prediction and state control
from predict_landmark_model import (
    predict_landmark,
    reset_text as reset_landmark_text,
    backspace_text as backspace_landmark_text,
    get_current_text as get_landmark_text
)
from predict_hybrid_model import (
    predict_hybrid,
    reset_text as reset_hybrid_text,
    backspace_text as backspace_hybrid_text,
    get_current_text as get_hybrid_text
)

app = Flask(__name__)
mode_selected = "landmark"
latest_text = ""

@app.route("/")
def index():
    print("[INFO] Home page loaded")
    return render_template("index.html")



@app.route("/set_mode", methods=["POST"])
def set_mode():
    global mode_selected
    mode_selected = request.form.get("mode")
    print(f"[INFO] Mode changed to: {mode_selected}")
    return "Mode updated"



def gen_frames():
    global latest_text
    print("[INFO] Camera feed initiated")
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("[ERROR] External camera (1) not accessible. Trying camera 0.")
        cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[CRITICAL ERROR] No camera accessible. Video feed will not start.")
        # Yield an empty frame or error message to the client
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + b"\r\n") # Empty frame
        return # Exit generator if no camera

    cap.set(3, 960)
    cap.set(4, 480)
    frame_count = 0
    
    while True:
        success, frame = cap.read()
        if not success:
            print(f"[ERROR] Failed to read from camera at frame {frame_count}. Retrying...")
            time.sleep(0.1) # Small delay before retrying
            continue
        frame = cv2.flip(frame, 1)
        frame_count += 1
        # print(f"[DEBUG] Processing frame {frame_count}") # Verbose logging for debugging

        if mode_selected == "hybrid":
            processed_frame, current_predicted_text = predict_hybrid(frame)
        else:
            processed_frame, current_predicted_text = predict_landmark(frame)

        # Update the global latest_text for the SSE stream
        if latest_text != current_predicted_text: # Only update if text has changed
            latest_text = current_predicted_text
            # print(f"[DEBUG] latest_text updated to: '{latest_text}'")

        try:
            # Ensure processed_frame is not None or empty before encoding
            if processed_frame is None or processed_frame.size == 0:
                print(f"[ERROR] Processed frame is empty or None at frame {frame_count}. Skipping.")
                continue

            _, buffer = cv2.imencode(".jpg", processed_frame)
            if buffer is None:
                print(f"[ERROR] Failed to encode frame {frame_count} to JPG. Skipping.")
                continue

            yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buffer.tobytes() + b"\r\n")
        except Exception as e:
            print(f"[CRITICAL ERROR] Error during frame encoding/yielding at frame {frame_count}: {e}")
            # Consider breaking here if errors are persistent, or log and continue
            break # Break if encoding consistently fails

    cap.release()
    print("[INFO] Camera feed closed")
    
    

@app.route("/video_feed")
def video_feed():
    print("[INFO] /video_feed route hit")
    return Response(gen_frames(), mimetype="multipart/x-mixed-replace; boundary=frame")



@app.route("/text_stream")
def text_stream():
    print("[INFO] /text_stream endpoint hit")
    def stream():
        global latest_text
        while True:
            try: # ADDED: Try-except block for robust streaming
                # print(f"[STREAM] sending: {latest_text.strip()}") # Keep this for debugging if needed
                time.sleep(0.5)
                yield f"data:{latest_text.strip() if latest_text else ''}\n\n"
            except Exception as e: # ADDED: Catching exceptions
                print(f"[STREAM ERROR] Error in text_stream generator: {e}")
                yield f"data:ERROR: {e}\n\n" # Inform client of error
                break # Terminate stream on error
    return Response(stream_with_context(stream()), mimetype="text/event-stream")


#starting capture code without any edit it was ------
# @app.route("/capture", methods=["POST"])
# def capture():
#     word = request.form.get("word")
#     print(f"[INFO] Capture requested for word: '{word}'")
#     if word:
#         # ORIGINAL LOGIC: Directly calling imported functions
#         # These functions now handle their own camera capture and saving
#         capture_landmark(word, mode="landmark")
#         capture_hybrid(word, mode="hybrid")
#         print(f"[INFO] Gesture '{word}' captured for both landmark and hybrid datasets.")
#     return redirect(url_for("index"))


# capture_mode = False
# capturing_word = ""

@app.route("/capture", methods=["POST"])
def capture():
    word = request.form.get("word")
    if not word:
        return redirect(url_for("index"))

    word = word.strip().capitalize()
    landmark_path = f"dataset_landmark/word_landmark/{word}"
    hybrid_path = f"dataset_hybrid/word_hybrid/{word}"

    landmark_exists = os.path.exists(landmark_path)
    hybrid_exists = os.path.exists(hybrid_path)

    if landmark_exists and hybrid_exists:
        choice = request.form.get("choice")  # frontend sends 'overwrite' or 'exit'
        if choice == "exit":
            print(f"[INFO] User exited capture for '{word}'")
            return redirect(url_for("index"))
        
        
        elif choice == "overwrite":
            # Clear landmark
            for file in os.listdir(landmark_path):
                # try:
                #     os.remove(os.path.join(landmark_path, file))
                # except Exception as e:
                    # print(f"[ERROR] Could not delete landmark file: {e}")
                
                # ✅ Replace entire landmark clear block with:
                try:
                    shutil.rmtree(landmark_path)
                    print(f"[INFO] Cleared landmark path for '{word}'")
                except Exception as e:
                    print(f"[ERROR] Could not clear landmark directory: {e}")
                                
                    
            # Clear hybrid
            for file in os.listdir(hybrid_path):
                file_path = os.path.join(hybrid_path, file)
                if os.path.isfile(file_path):
                    # ✅ Replace entire landmark clear block with:
                    try:
                        shutil.rmtree(hybrid_path)
                        print(f"[INFO] Cleared landmark path for '{word}'")
                    except Exception as e:
                        print(f"[ERROR] Could not clear landmark directory: {e}")
                        
                    # try:
                    #     os.remove(file_path)
                    # except Exception as e:
                    #     print(f"[ERROR] Could not delete hybrid file: {e}")
                
                
                
            # Safely clear preview images
            preview_path = os.path.join(hybrid_path, "preview")
            if os.path.exists(preview_path):
                for img in os.listdir(preview_path):
                    img_path = os.path.join(preview_path, img)
                    try:
                        os.remove(img_path)
                    except Exception as e:
                        print(f"[ERROR] Failed to delete '{img_path}': {e}")

            print(f"[INFO] Overwrote existing data for '{word}'")

    # Ensure dirs exist
    if not os.path.exists(landmark_path):
        os.makedirs(landmark_path, exist_ok=True)
    if not os.path.exists(hybrid_path):
        os.makedirs(hybrid_path, exist_ok=True)
    if not os.path.exists(os.path.join(hybrid_path, "preview")):
        os.makedirs(os.path.join(hybrid_path, "preview"), exist_ok=True)

        

    # capture_landmark(word, mode="landmark")
    # capture_hybrid(word, mode="hybrid")
    
    # global capture_mode, capturing_word
    # capture_mode = True
    # capturing_word = word
    # ✅ Launch landmark + hybrid capture together
    # subprocess.Popen(["python", "capture_landmark_data.py", word])
    # subprocess.Popen(["python", "capture_hybrid_data.py", word])
    # subprocess.Popen(["python", "capture_both.py", word])

    # print(f"[INFO] Gesture '{word}' captured for both landmark and hybrid datasets.")
    # return redirect(url_for("index"))
    # return "OK"
    print(f"[INFO] Starting capture stream for '{word}'")
    # return redirect(url_for("capture_feed", word=word))
    return redirect(url_for("capture_ui", word=word))



@app.route("/capture_feed")
def capture_feed():
    word = request.args.get("word")
    # return Response(capture_stream(word), mimetype="multipart/x-mixed-replace; boundary=frame")
    # return Response(capture_generator(word), mimetype="multipart/x-mixed-replace; boundary=frame")
    return Response(capture_both.capture_stream(word, capturing_status), mimetype="multipart/x-mixed-replace; boundary=frame")

    # def gen():
    #     # global capturing_done
    #     # capturing_done = False
    #     for frame in capture_generator(word):  # yields frames
    #         # if capturing_done:
    #         #     break
    #         yield (b"--frame\r\n"
    #                b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    #     print(f"[INFO] Capture stream ended for '{word}'")
    # return Response(gen(), mimetype="multipart/x-mixed-replace; boundary=frame")



@app.route("/check_word_exists", methods=["POST"])
def check_word_exists():
    word = request.form.get("word").strip().capitalize()
    landmark_path = f"dataset_landmark/word_landmark/{word}"
    hybrid_path = f"dataset_hybrid/word_hybrid/{word}"

    if os.path.exists(landmark_path) and os.path.exists(hybrid_path):
        return "both_exist"
    return "new_or_partial"



@app.route("/train", methods=["POST"])
def train():
    global training_process
    mode = request.form.get("mode", mode_selected)
    if os.path.exists("training_progress.log"):
        os.remove("training_progress.log")  # Clear old progress
    if mode == "hybrid":
        training_process = subprocess.Popen(["python", "train_hybrid_model.py"])
    else:
        training_process = subprocess.Popen(["python", "train_landmark_model.py"])
    return "Training started!"


# # for capture stopping --
# @app.route("/stop_training", methods=["POST"])
# def stop_training():
#     global training_process
#     if training_process and training_process.poll() is None:
#         training_process.terminate()
#         training_process = None
#         print("[INFO] Training process was manually stopped by user.")
#         return "Training stopped"
#     return "No active training"



# @app.route("/training_progress")
# def training_progress():
#     try:
#         with open("training_progress.log", "r") as f:
#             content = f.read().strip()
#             if not content:
#                 return "0"
#             progress = int(content)
#             if progress >= 100 and training_process:
#                 training_process.poll()  # Check if process ended
#             return str(progress)
#     except:
#         return "0"


@app.route("/training_progress")
def training_progress():
    try:
        with open("training_progress.log", "r") as f:
            content = f.read().strip()
            if not content:
                return "0"
            progress = int(content)

            # ⏩ ADD THIS LOGIC
            if progress >= 100:
                training_progress.incomplete_count = 0  # Reset counter after complete
                return str(progress)

            # Count consecutive incomplete calls
            if not hasattr(training_progress, 'incomplete_count'):
                training_progress.incomplete_count = 0
            training_progress.incomplete_count += 1

            if training_progress.incomplete_count >= 10:
                print("[INFO] Stopping progress responses after 10 incomplete checks.")
                return "STOP"

            return str(progress)
    except:
        return "0"




# @app.route("/train", methods=["POST"])
# def train():
#     os.remove("training_process.log")
#     mode = request.form.get("mode", mode_selected)
#     print(f"[INFO] Training started for mode: {mode}")
#     if mode == "hybrid":
#         training_process = subprocess.Popen(["python", "train_hybrid_model.py"])
#     else:
#         training_process = subprocess.Popen(["python", "train_landmark_model.py"])
#     return "Training started!"


# @app.route("/training_process")
# def training_process():
#     try:
#         with open("training_progress.log","r") as f:
#             content = f.read().strip()
#             if not content:
#                 return "0"
#             progress = int(content)
#             if progress >= 100 and training_process:
#                 training_process.poll()
#             return str(progress)
#     except:
#         return "0"



# @app.route("/progress")
# def progress():
#     try:
#         with open("training_progress.log", "r") as f:
#             return f.read().strip() # ORIGINAL LOGIC: Returns raw content
#     except:
#         return "0"



@app.route("/reset_text", methods=["POST"])
def reset_text():
    global latest_text
    print(f"[INFO] /reset_text POST hit, mode: {mode_selected}")
    if mode_selected == "landmark":
        reset_landmark_text() # This now directly modifies typed_text in predict_landmark_model.py
        latest_text = get_landmark_text() # Get the immediately updated text
    else:
        reset_hybrid_text() # This now directly modifies typed_text in predict_hybrid_model.py
        latest_text = get_hybrid_text() # Get the immediately updated text
    print(f"[INFO] Text after reset: '{latest_text}'")
    return latest_text # Return the updated text to the frontend




@app.route("/backspace_text", methods=["POST"])
def backspace_text():
    global latest_text
    print(f"[INFO] /backspace_text POST hit, mode: {mode_selected}")
    if mode_selected == "landmark":
        backspace_landmark_text() # This now directly modifies typed_text in predict_landmark_model.py
        latest_text = get_landmark_text() # Get the immediately updated text
    else:
        backspace_hybrid_text() # This now directly modifies typed_text in predict_hybrid_model.py
        latest_text = get_hybrid_text() # Get the immediately updated text
    print(f"[INFO] Text after backspace: '{latest_text}'")
    return latest_text # Return the updated text to the frontend




@app.route("/download_dataset")
def download_dataset():
    mode = request.args.get("mode", "landmark")
    print(f"[INFO] Dataset download requested for mode: {mode}")
    base_path = f"dataset_{mode}/word_{mode}"
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w') as zipf:
        for word in os.listdir(base_path):
            preview_path = os.path.join(base_path, word, "preview")
            if os.path.exists(preview_path):
                for img_file in os.listdir(preview_path):
                    if img_file.endswith(".jpg"):
                        full_path = os.path.join(preview_path, img_file)
                        arcname = f"{word}/{img_file}"
                        zipf.write(full_path, arcname)
    zip_buffer.seek(0)
    return send_file(zip_buffer, mimetype='application/zip', as_attachment=True, download_name='dataset.zip')



# @app.route("/test_capture")
# def test_capture():
#     return render_template("test_capture.html")

# capturing_status = {}

@app.route("/capture_ui")
def capture_ui():
    word = request.args.get("word")
    # return render_template("capture_ui.html", word=word)
    timestamp = int(time.time())  #Add current timestamp 
    return render_template("capture_ui.html", word=word, timestamp=timestamp)


@app.route("/capture_status")
def capture_status():
    # word = request.args.get("word")
    # return capturing_status.get(word, "capturing")
    word = request.args.get("word")
    if request.args.get("interrupt") == "1":
        capture_both.speak(f"Capture interrupted for {word}.") # Speak on interruption
        capturing_status[word] = "interrupted"
        return "interrupted"
    return capturing_status.get(word, "capturing")



if __name__ == "__main__":
    print("[INFO] Flask app started on http://127.0.0.1:5000")
    app.run(debug=False)

