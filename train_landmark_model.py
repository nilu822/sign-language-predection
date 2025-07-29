# train_landmark_model.py

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import absl.logging
absl.logging.set_verbosity(absl.logging.FATAL)

import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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

# === Epoch tracking for live UI ===
LOG_FILE = "training_progress.log"

class EpochLogger(tf.keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        with open(LOG_FILE, "w") as f:
            f.write("0")  # Reset progress at start

    def on_epoch_end(self, epoch, logs=None):
        progress = int(((epoch + 1) / self.params['epochs']) * 100)
        with open(LOG_FILE, "w") as f:
            f.write(str(progress))

    def on_train_end(self, logs=None):
        with open(LOG_FILE, "w") as f:
            f.write("100")  # Ensure final 100%

# === Config ===
DATASET_PATH = 'dataset_landmark/word_landmark'
MODEL_DIR = 'model_landmark'
CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoint')
MODEL_H5_PATH = os.path.join(MODEL_DIR, 'model_landmark.h5')
MODEL_KERAS_PATH = os.path.join(MODEL_DIR, 'sign_model_landmark.keras')
LABEL_PATH = os.path.join(MODEL_DIR, 'labels.txt')
CONF_MATRIX_PATH = os.path.join(MODEL_DIR, 'confusion_matrix.png')

EPOCHS = 30
BATCH_SIZE = 32
INPUT_SHAPE = (42,)

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# === Load Dataset ===
def load_data():
    X, y, class_names = [], [], []
    print("[INFO]:- Loading dataset...")

    for idx, class_name in enumerate(sorted(os.listdir(DATASET_PATH))):
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.isdir(class_path):
            continue
        class_names.append(class_name)
        files = [f for f in os.listdir(class_path) if f.endswith(".npy")]
        print(f"Class '{class_name}': {len(files)} samples")
        for file in files:
            path = os.path.join(class_path, file)
            X.append(np.load(path))
            y.append(idx)

    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=len(class_names))

    print(f"[INFO]:- Total classes loaded: {len(class_names)}")
    return X, y, class_names

# === Build Model ===
def build_model(num_classes):
    model = Sequential([
        Dense(128, activation='relu', input_shape=INPUT_SHAPE),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()
    return model

X, y, class_names = load_data()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = build_model(len(class_names))

checkpoint_cb = ModelCheckpoint(os.path.join(CHECKPOINT_DIR, 'model_checkpoint.keras'),
                                save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

start_time = datetime.now()
speak("Training started")
print(f"\n[INFO]:- Training started at --> {start_time.strftime('%H:%M:%S')}\n")

history = model.fit(X_train, y_train,
                    validation_data=(X_test, y_test),
                    epochs=EPOCHS,
                    batch_size=BATCH_SIZE,
                    callbacks=[checkpoint_cb, early_stop, reduce_lr, EpochLogger()],
                    shuffle=True)

val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"[INFO]:- Final Validation Accuracy ---> {val_acc:.4f}")

y_pred_probs = model.predict(X_test)
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












# This was working -- with fake progress bar ------ correct it was
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# import absl.logging
# absl.logging.set_verbosity(absl.logging.FATAL)

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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

# # === Epoch tracking for live UI ===
# LOG_FILE = "training_progress.log"

# class EpochLogger(tf.keras.callbacks.Callback):
#     def on_epoch_end(self, epoch, logs=None):
#         with open(LOG_FILE, "w") as f:
#             f.write(str(epoch + 1))

# # === Config ===
# DATASET_PATH = 'dataset_landmark/word_landmark'
# MODEL_DIR = 'model_landmark'
# CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoint')
# MODEL_H5_PATH = os.path.join(MODEL_DIR, 'model_landmark.h5')
# MODEL_KERAS_PATH = os.path.join(MODEL_DIR, 'sign_model_landmark.keras')
# LABEL_PATH = os.path.join(MODEL_DIR, 'labels.txt')
# CONF_MATRIX_PATH = os.path.join(MODEL_DIR, 'confusion_matrix.png')
# METRICS_PLOT_PATH = os.path.join(MODEL_DIR, 'training_metrics.png')

# EPOCHS = 30
# BATCH_SIZE = 32
# INPUT_SHAPE = (42,)

# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# # === Load Dataset ===
# def load_data():
#     X, y, class_names = [], [], []
#     print("[INFO]:- Loading dataset...")

#     for idx, class_name in enumerate(sorted(os.listdir(DATASET_PATH))):
#         class_path = os.path.join(DATASET_PATH, class_name)
#         if not os.path.isdir(class_path):
#             continue
#         class_names.append(class_name)
#         files = [f for f in os.listdir(class_path) if f.endswith(".npy")]
#         print(f"Class '{class_name}': {len(files)} samples")
#         for file in files:
#             path = os.path.join(class_path, file)
#             X.append(np.load(path))
#             y.append(idx)

#     X = np.array(X)
#     y = to_categorical(np.array(y), num_classes=len(class_names))

#     print(f"[INFO]:- Total classes loaded: {len(class_names)}")
#     return X, y, class_names

# # === Build Model ===
# def build_model(num_classes):
#     model = Sequential([
#         Dense(128, activation='relu', input_shape=INPUT_SHAPE),
#         Dropout(0.3),
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     model.summary()
#     return model

# X, y, class_names = load_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model = build_model(len(class_names))

# checkpoint_cb = ModelCheckpoint(os.path.join(CHECKPOINT_DIR, 'model_checkpoint.keras'),
#                                 save_best_only=True, monitor='val_accuracy', mode='max')
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# start_time = datetime.now()
# speak("Training started")
# print(f"\n[INFO]:- Training started at --> {start_time.strftime('%H:%M:%S')}\n")

# history = model.fit(X_train, y_train,
#                     validation_data=(X_test, y_test),
#                     epochs=EPOCHS,
#                     batch_size=BATCH_SIZE,
#                     callbacks=[checkpoint_cb, early_stop, reduce_lr, EpochLogger()],
#                     shuffle=True)

# val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
# print(f"[INFO]:- Final Validation Accuracy ---> {val_acc:.4f}")

# y_pred_probs = model.predict(X_test)
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
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning)
# import absl.logging
# absl.logging.set_verbosity(absl.logging.FATAL)

# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
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
# DATASET_PATH = 'dataset_landmark/word_landmark'
# MODEL_DIR = 'model_landmark'
# CHECKPOINT_DIR = os.path.join(MODEL_DIR, 'checkpoint')
# MODEL_H5_PATH = os.path.join(MODEL_DIR, 'model_landmark.h5')
# MODEL_KERAS_PATH = os.path.join(MODEL_DIR, 'sign_model_landmark.keras')
# LABEL_PATH = os.path.join(MODEL_DIR, 'labels.txt')
# CONF_MATRIX_PATH = os.path.join(MODEL_DIR, 'confusion_matrix.png')
# METRICS_PLOT_PATH = os.path.join(MODEL_DIR, 'training_metrics.png')

# EPOCHS = 30
# BATCH_SIZE = 32
# INPUT_SHAPE = (42,)

# os.makedirs(MODEL_DIR, exist_ok=True)
# os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# # === Load Dataset ===
# def load_data():
#     X, y, class_names = [], [], []
#     print("[INFO]:- Loading dataset...")

#     for idx, class_name in enumerate(sorted(os.listdir(DATASET_PATH))):
#         class_path = os.path.join(DATASET_PATH, class_name)
#         if not os.path.isdir(class_path):
#             continue
#         class_names.append(class_name)
#         files = [f for f in os.listdir(class_path) if f.endswith(".npy")]
#         print(f"Class '{class_name}': {len(files)} samples")
#         for file in files:
#             path = os.path.join(class_path, file)
#             X.append(np.load(path))
#             y.append(idx)

#     X = np.array(X)
#     y = to_categorical(np.array(y), num_classes=len(class_names))

#     print(f"[INFO]:- Total classes loaded: {len(class_names)}")
#     return X, y, class_names

# # === Build Model ===
# def build_model(num_classes):
#     model = Sequential([
#         Dense(128, activation='relu', input_shape=INPUT_SHAPE),
#         Dropout(0.3),
#         Dense(64, activation='relu'),
#         Dropout(0.3),
#         Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer='adam',
#                   loss='categorical_crossentropy',
#                   metrics=['accuracy'])
#     model.summary()
#     return model

# # === Load and prepare data ===
# X, y, class_names = load_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # === Prepare model and callbacks ===
# model = build_model(len(class_names))
# checkpoint_cb = ModelCheckpoint(os.path.join(CHECKPOINT_DIR, 'model_checkpoint.keras'),
#                                 save_best_only=True, monitor='val_accuracy', mode='max')
# early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
# reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# # === Start Training ===
# start_time = datetime.now()
# speak("Training started")
# print(f"\n[INFO]:- Training started at --> {start_time.strftime('%H:%M:%S')}\n")

# history = model.fit(X_train, y_train,
#                     validation_data=(X_test, y_test),
#                     epochs=EPOCHS,
#                     batch_size=BATCH_SIZE,
#                     callbacks=[checkpoint_cb, early_stop, reduce_lr],
#                     shuffle=True)

# # === Evaluate model ===
# val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
# print(f"[INFO]:- Final Validation Accuracy ---> {val_acc:.4f}")

# # === Predictions and metrics ===
# y_pred_probs = model.predict(X_test)
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

# # === Save model ===
# model.save(MODEL_H5_PATH)
# model.save(MODEL_KERAS_PATH)
# print(f"[INFO]:-  Models saved to:")
# print(f" • {MODEL_H5_PATH}")
# print(f" • {MODEL_KERAS_PATH}")

# # === Save labels ===
# with open(LABEL_PATH, "w") as f:
#     for label in class_names:
#         f.write(label + "\n")
# print(f"[INFO]:- Labels saved to --> {LABEL_PATH}")

# # === Plot training curves ===
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

# final_accuracy = val_acc * 100
# print(f"[INFO]:- Final Validation Accuracy ----> {final_accuracy:.2f}%")

# play_beep()
# speak("Training completed. All files saved successfully to their respective directories")
