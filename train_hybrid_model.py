# # train_hybrid_model.py
# train_hybrid_model.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import absl.logging
absl.logging.set_verbosity(absl.logging.FATAL)

import numpy as np
import tensorflow as tf
import cv2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, concatenate
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pyttsx3
import winsound

# TTS setup
tts = pyttsx3.init()
tts.setProperty('rate', 160)
tts.setProperty('voice', tts.getProperty('voices')[1].id)

def speak(text):
    try:
        tts.say(text)
        tts.runAndWait()
    except Exception as e:
        print(f"[TTS ERROR]: {e}")

def play_beep():
    winsound.Beep(1000, 300)

# === Epoch progress log ===
LOG_FILE = "training_progress.log"

class EpochLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        with open(LOG_FILE, "w") as f:
            f.write("0")

    def on_epoch_end(self, epoch, logs=None):
        progress = int(((epoch + 1) / self.params['epochs']) * 100)
        with open(LOG_FILE, "w") as f:
            f.write(str(progress))

    def on_train_end(self, logs=None):
        with open(LOG_FILE, "w") as f:
            f.write("100")

# === Config ===
DATASET_PATH = 'dataset_hybrid/word_hybrid'
MODEL_DIR = 'model_hybrid'
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoint')
MODEL_H5_PATH = os.path.join(MODEL_DIR, 'model_hybrid.h5')
MODEL_KERAS_PATH = os.path.join(MODEL_DIR, 'sign_model_hybrid.keras')
LABEL_PATH = os.path.join(MODEL_DIR, 'labels.txt')
CONF_MATRIX_PATH = os.path.join(MODEL_DIR, 'confusion_matrix.png')

EPOCHS = 30
BATCH_SIZE = 32
IMAGE_SIZE = (64, 64)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def load_data():
    X_landmark, X_image, y, class_names = [], [], [], []
    print("[INFO]:- Loading hybrid dataset...")

    for idx, class_name in enumerate(sorted(os.listdir(DATASET_PATH))):
        class_path = os.path.join(DATASET_PATH, class_name)
        preview_path = os.path.join(class_path, "preview")
        if not os.path.isdir(class_path) or not os.path.exists(preview_path):
            continue
        class_names.append(class_name)
        files = sorted([f for f in os.listdir(class_path) if f.endswith(".npy")])
        print(f"Class '{class_name}': {len(files)} samples")

        for file in files:
            npy_path = os.path.join(class_path, file)
            jpg_path = os.path.join(preview_path, file.replace('.npy', '.jpg'))

            if not os.path.exists(jpg_path):
                continue

            landmark = np.load(npy_path)
            if landmark.shape != (42,):
                continue
            X_landmark.append(landmark)

            image = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
            image = cv2.resize(image, IMAGE_SIZE)
            X_image.append(image)

            y.append(idx)

    X_landmark = np.array(X_landmark)
    X_image = np.expand_dims(np.array(X_image), axis=-1) / 255.0
    y = to_categorical(np.array(y), num_classes=len(class_names))

    print(f"[INFO]:- Total classes loaded: {len(class_names)}")
    return X_landmark, X_image, y, class_names

def build_model(num_classes):
    input_landmark = Input(shape=(42,), name='landmark_input')
    x1 = Dense(128, activation='relu')(input_landmark)
    x1 = Dropout(0.3)(x1)

    input_image = Input(shape=(64, 64, 1), name='image_input')
    x2 = Conv2D(32, (3, 3), activation='relu')(input_image)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Conv2D(64, (3, 3), activation='relu')(x2)
    x2 = MaxPooling2D((2, 2))(x2)
    x2 = Flatten()(x2)

    combined = concatenate([x1, x2])
    x = Dense(64, activation='relu')(combined)
    x = Dropout(0.3)(x)
    output = Dense(num_classes, activation='softmax')(x)

    model = Model(inputs=[input_landmark, input_image], outputs=output)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

X_landmark, X_image, y, class_names = load_data()
X_train_lm, X_test_lm, X_train_img, X_test_img, y_train, y_test = train_test_split(
    X_landmark, X_image, y, test_size=0.2, random_state=42)

model = build_model(len(class_names))

checkpoint_cb = ModelCheckpoint(os.path.join(CHECKPOINT_DIR, 'model_checkpoint.keras'),
                                save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

start_time = datetime.now()
speak("Training started")
print(f"\n[INFO]:- Training started at --> {start_time.strftime('%H:%M:%S')}\n")

history = model.fit([X_train_lm, X_train_img], y_train,
                    validation_data=([X_test_lm, X_test_img], y_test),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[checkpoint_cb, early_stop, reduce_lr, EpochLogger()],
                    shuffle=True)

val_loss, val_acc = model.evaluate([X_test_lm, X_test_img], y_test, verbose=0)
print(f"[INFO]:- Final Validation Accuracy ---> {val_acc:.4f}")

y_pred_probs = model.predict([X_test_lm, X_test_img])
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n[INFO]:- Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig(CONF_MATRIX_PATH)

model.save(MODEL_H5_PATH)
model.save(MODEL_KERAS_PATH)

with open(LABEL_PATH, "w") as f:
    for label in class_names:
        f.write(label + "\n")

print(f"[INFO]:- Model, confusion matrix, and labels saved successfully.")
play_beep()
speak("Training completed. All files saved successfully to their respective directories")













# this was the working code -----
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# import absl.logging
# absl.logging.set_verbosity(absl.logging.FATAL)

# import numpy as np
# import tensorflow as tf
# import cv2
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, concatenate
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import pyttsx3
# import winsound

# # TTS setup
# tts = pyttsx3.init()
# tts.setProperty('rate', 160)
# tts.setProperty('voice', tts.getProperty('voices')[1].id)

# def speak(text):
#     try:
#         tts.say(text)
#         tts.runAndWait()
#     except Exception as e:
#         print(f"[TTS ERROR]: {e}")

# def play_beep():
#     winsound.Beep(1000, 300)

# # === Epoch progress log ===
# LOG_FILE = "training_progress.log"

# class EpochLogger(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         with open(LOG_FILE, "w") as f:
#             f.write(str(epoch + 1))

# # === Config ===
# DATASET_PATH = 'dataset_hybrid/word_hybrid'
# MODEL_DIR = 'model_hybrid'
# CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoint')
# MODEL_H5_PATH = os.path.join(MODEL_DIR, 'model_hybrid.h5')
# MODEL_KERAS_PATH = os.path.join(MODEL_DIR, 'sign_model_hybrid.keras')
# LABEL_PATH = os.path.join(MODEL_DIR, 'labels.txt')
# CONF_MATRIX_PATH = os.path.join(MODEL_DIR, 'confusion_matrix.png')
# METRICS_PLOT_PATH = os.path.join(MODEL_DIR, 'training_metrics.png')

# EPOCHS = 30
# BATCH_SIZE = 32
# IMAGE_SIZE = (64, 64)

# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# def load_data():
#     X_landmark, X_image, y, class_names = [], [], [], []
#     print("[INFO]:- Loading hybrid dataset...")

#     for idx, class_name in enumerate(sorted(os.listdir(DATASET_PATH))):
#         class_path = os.path.join(DATASET_PATH, class_name)
#         preview_path = os.path.join(class_path, "preview")
#         if not os.path.isdir(class_path) or not os.path.exists(preview_path):
#             continue
#         class_names.append(class_name)
#         files = sorted([f for f in os.listdir(class_path) if f.endswith(".npy")])
#         print(f"Class '{class_name}': {len(files)} samples")

#         for file in files:
#             npy_path = os.path.join(class_path, file)
#             jpg_path = os.path.join(preview_path, file.replace('.npy', '.jpg'))

#             if not os.path.exists(jpg_path):
#                 continue

#             landmark = np.load(npy_path)
#             if landmark.shape != (42,):
#                 continue
#             X_landmark.append(landmark)

#             image = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
#             image = cv2.resize(image, IMAGE_SIZE)
#             X_image.append(image)

#             y.append(idx)

#     X_landmark = np.array(X_landmark)
#     X_image = np.expand_dims(np.array(X_image), axis=-1) / 255.0
#     y = to_categorical(np.array(y), num_classes=len(class_names))

#     print(f"[INFO]:- Total classes loaded: {len(class_names)}")
#     return X_landmark, X_image, y, class_names

# def build_model(num_classes):
#     input_landmark = Input(shape=(42,), name='landmark_input')
#     x1 = Dense(128, activation='relu')(input_landmark)
#     x1 = Dropout(0.3)(x1)

#     input_image = Input(shape=(64, 64, 1), name='image_input')
#     x2 = Conv2D(32, (3, 3), activation='relu')(input_image)
#     x2 = MaxPooling2D((2, 2))(x2)
#     x2 = Conv2D(64, (3, 3), activation='relu')(x2)
#     x2 = MaxPooling2D((2, 2))(x2)
#     x2 = Flatten()(x2)

#     combined = concatenate([x1, x2])
#     x = Dense(64, activation='relu')(combined)
#     x = Dropout(0.3)(x)
#     output = Dense(num_classes, activation='softmax')(x)

#     model = Model(inputs=[input_landmark, input_image], outputs=output)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     model.summary()
#     return model

# X_landmark, X_image, y, class_names = load_data()
# X_train_lm, X_test_lm, X_train_img, X_test_img, y_train, y_test = train_test_split(
#     X_landmark, X_image, y, test_size=0.2, random_state=42)

# model = build_model(len(class_names))

# checkpoint_cb = ModelCheckpoint(os.path.join(CHECKPOINT_DIR, 'model_checkpoint.keras'),
#                                 save_best_only=True, monitor='val_accuracy', mode='max')
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# start_time = datetime.now()
# speak("Training started")
# print(f"\n[INFO]:- Training started at --> {start_time.strftime('%H:%M:%S')}\n")

# history = model.fit([X_train_lm, X_train_img], y_train,
#                     validation_data=([X_test_lm, X_test_img], y_test),
#                     epochs=EPOCHS,
#                     batch_size=BATCH_SIZE,
#                     callbacks=[checkpoint_cb, early_stop, reduce_lr, EpochLogger()],
#                     shuffle=True)

# val_loss, val_acc = model.evaluate([X_test_lm, X_test_img], y_test, verbose=0)
# print(f"[INFO]:- Final Validation Accuracy ---> {val_acc:.4f}")

# y_pred_probs = model.predict([X_test_lm, X_test_img])
# y_pred = np.argmax(y_pred_probs, axis=1)
# y_true = np.argmax(y_test, axis=1)

# print("\n[INFO]:- Classification Report:")
# print(classification_report(y_true, y_pred, target_names=class_names))

# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.savefig(CONF_MATRIX_PATH)

# model.save(MODEL_H5_PATH)
# model.save(MODEL_KERAS_PATH)

# with open(LABEL_PATH, "w") as f:
#     for label in class_names:
#         f.write(label + "\n")

# print(f"[INFO]:- Model, confusion matrix, and labels saved successfully.")
# play_beep()
# speak("Training completed. All files saved successfully to their respective directories")



















# this is for the backend ---- was using this --- original 
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# import absl.logging
# absl.logging.set_verbosity(absl.logging.FATAL)

# import numpy as np
# import tensorflow as tf
# import cv2
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Input, Dense, Dropout, Conv2D, MaxPooling2D, Flatten, concatenate
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import pyttsx3
# import winsound

# # === TTS setup ===
# tts = pyttsx3.init()
# tts.setProperty('rate', 160)
# tts.setProperty('voice', tts.getProperty('voices')[1].id)

# def speak(text):
#     tts.say(text)
#     tts.runAndWait()

# def play_beep():
#     winsound.Beep(1000, 300)

# # === Config ===
# DATASET_PATH = 'dataset_hybrid/word_hybrid'
# MODEL_DIR = 'model_hybrid'
# CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoint')
# MODEL_H5_PATH = os.path.join(MODEL_DIR, 'model_hybrid.h5')
# MODEL_KERAS_PATH = os.path.join(MODEL_DIR, 'sign_model_hybrid.keras')
# LABEL_PATH = os.path.join(MODEL_DIR, 'labels.txt')
# CONF_MATRIX_PATH = os.path.join(MODEL_DIR, 'confusion_matrix.png')
# METRICS_PLOT_PATH = os.path.join(MODEL_DIR, 'training_metrics.png')

# EPOCHS = 30
# BATCH_SIZE = 32
# IMAGE_SIZE = (64, 64)

# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# # === Load Dataset ===
# def load_data():
#     X_landmark, X_image, y, class_names = [], [], [], []
#     print("[INFO]:- Loading hybrid dataset...")

#     for idx, class_name in enumerate(sorted(os.listdir(DATASET_PATH))):
#         class_path = os.path.join(DATASET_PATH, class_name)
#         preview_path = os.path.join(class_path, "preview")
#         if not os.path.isdir(class_path) or not os.path.exists(preview_path):
#             continue
#         class_names.append(class_name)
#         files = sorted([f for f in os.listdir(class_path) if f.endswith(".npy")])
#         print(f"Class '{class_name}': {len(files)} samples")

#         for file in files:
#             npy_path = os.path.join(class_path, file)
#             jpg_path = os.path.join(preview_path, file.replace('.npy', '.jpg'))

#             if not os.path.exists(jpg_path):
#                 continue

#             # Load landmark
#             landmark = np.load(npy_path)
#             if landmark.shape != (42,):
#                 continue
#             X_landmark.append(landmark)

#             # Load image (grayscale, resize)
#             image = cv2.imread(jpg_path, cv2.IMREAD_GRAYSCALE)
#             image = cv2.resize(image, IMAGE_SIZE)
#             X_image.append(image)

#             y.append(idx)

#     X_landmark = np.array(X_landmark)
#     X_image = np.expand_dims(np.array(X_image), axis=-1) / 255.0
#     y = to_categorical(np.array(y), num_classes=len(class_names))

#     print(f"[INFO]:- Total classes loaded: {len(class_names)}")
#     return X_landmark, X_image, y, class_names

# # === Build Hybrid Model ===
# def build_model(num_classes):
#     # Landmark input
#     input_landmark = Input(shape=(42,), name='landmark_input')
#     x1 = Dense(128, activation='relu')(input_landmark)
#     x1 = Dropout(0.3)(x1)

#     # Image input
#     input_image = Input(shape=(64, 64, 1), name='image_input')
#     x2 = Conv2D(32, (3, 3), activation='relu')(input_image)
#     x2 = MaxPooling2D((2, 2))(x2)
#     x2 = Conv2D(64, (3, 3), activation='relu')(x2)
#     x2 = MaxPooling2D((2, 2))(x2)
#     x2 = Flatten()(x2)

#     # Combine
#     combined = concatenate([x1, x2])
#     x = Dense(64, activation='relu')(combined)
#     x = Dropout(0.3)(x)
#     output = Dense(num_classes, activation='softmax')(x)

#     model = Model(inputs=[input_landmark, input_image], outputs=output)
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     model.summary()
#     return model

# # === Load Data ===
# X_landmark, X_image, y, class_names = load_data()
# X_train_lm, X_test_lm, X_train_img, X_test_img, y_train, y_test = train_test_split(
#     X_landmark, X_image, y, test_size=0.2, random_state=42)

# # === Build Model ===
# model = build_model(len(class_names))

# # === Callbacks ===
# checkpoint_cb = ModelCheckpoint(os.path.join(CHECKPOINT_DIR, 'model_checkpoint.keras'),
#                                 save_best_only=True, monitor='val_accuracy', mode='max')
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# # === Training ===
# start_time = datetime.now()
# speak("Training started")
# print(f"\n[INFO]:- Training started at --> {start_time.strftime('%H:%M:%S')}\n")

# history = model.fit([X_train_lm, X_train_img], y_train,
#                     validation_data=([X_test_lm, X_test_img], y_test),
#                     epochs=EPOCHS,
#                     batch_size=BATCH_SIZE,
#                     callbacks=[checkpoint_cb, early_stop, reduce_lr],
#                     shuffle=True)

# # === Evaluation ===
# val_loss, val_acc = model.evaluate([X_test_lm, X_test_img], y_test, verbose=0)
# print(f"[INFO]:- Final Validation Accuracy ---> {val_acc:.4f}")

# # === Predictions and Metrics ===
# y_pred_probs = model.predict([X_test_lm, X_test_img])
# y_pred = np.argmax(y_pred_probs, axis=1)
# y_true = np.argmax(y_test, axis=1)

# print("\n[INFO]:- Classification Report:")
# print(classification_report(y_true, y_pred, target_names=class_names))

# cm = confusion_matrix(y_true, y_pred)
# plt.figure(figsize=(8, 6))
# sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
# plt.xlabel("Predicted Label")
# plt.ylabel("True Label")
# plt.title("Confusion Matrix")
# plt.tight_layout()
# plt.savefig(CONF_MATRIX_PATH)
# print(f"[INFO]:- Confusion matrix saved to --> {CONF_MATRIX_PATH}")

# # === Save Models ===
# model.save(MODEL_H5_PATH)
# model.save(MODEL_KERAS_PATH)
# print(f"[INFO]:-  Models saved to:")
# print(f" • {MODEL_H5_PATH}")
# print(f" • {MODEL_KERAS_PATH}")

# # === Save Labels ===
# with open(LABEL_PATH, "w") as f:
#     for label in class_names:
#         f.write(label + "\n")
# print(f"[INFO]:- Labels saved to --> {LABEL_PATH}")

# # === Plot Training Curves ===
# plt.figure(figsize=(14, 5))

# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
# plt.plot(history.history['val_accuracy'], label='Val Accuracy', marker='o')
# plt.title('Accuracy Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# plt.legend()
# plt.grid(True)

# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Train Loss', marker='o')
# plt.plot(history.history['val_loss'], label='Val Loss', marker='o')
# plt.title('Loss Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.savefig(METRICS_PLOT_PATH)
# print(f"[INFO]:- Training metrics plot saved to --> {METRICS_PLOT_PATH}")

# # === End Training ===
# end_time = datetime.now()
# duration = end_time - start_time
# print(f"\n[INFO]:- Training ended at --> {end_time.strftime('%H:%M:%S')}")
# print(f"[INFO]:- Total training duration: {str(duration).split('.')[0]}")
# print(f"[INFO]:- Final Validation Accuracy ----> {val_acc * 100:.2f}%")

# play_beep()
# speak("Training completed. All files saved successfully to their respective directories")
