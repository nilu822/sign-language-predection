# added some features like MobileNetV2 etc for testing.. this is not original just for testing ---

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import absl.logging
absl.logging.set_verbosity(absl.logging.FATAL)

import cv2
import numpy as np
from datetime import datetime
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import class_weight
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pyttsx3
import winsound

# === TTS setup ===
tts = pyttsx3.init()
tts.setProperty('rate', 160)
tts.setProperty('voice', tts.getProperty('voices')[1].id)

def speak(text):
    tts.say(text)
    tts.runAndWait()

def play_beep():
    winsound.Beep(1000, 300)

# === Config ===
DATASET_PATH = 'dataset'
MODEL_H5_PATH = 'model/model.h5'
MODEL_KERAS_PATH = 'model/sign_model.keras'
CHECKPOINT_PATH = 'checkpoint/model_checkpoint.keras'
IMAGE_SIZE = 96
EPOCHS = 30
BATCH_SIZE = 32

# === Load Dataset ===
def load_data():
    print("[INFO] Loading dataset...")
    images, labels, class_names = [], [], []

    for folder_type in ['alphabet', 'word']:
        folder_path = os.path.join(DATASET_PATH, folder_type)
        if not os.path.exists(folder_path):
            continue
        # for class_name in sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]):        # was using this 
        for class_name in sorted([d for d in os.listdir(folder_path)
                          if os.path.isdir(os.path.join(folder_path, d)) and
                          len(os.listdir(os.path.join(folder_path, d))) > 0]):
            
            class_path = os.path.join(folder_path, class_name)
            if class_name not in class_names:
                class_names.append(class_name)
            label = class_names.index(class_name)
            for img_file in os.listdir(class_path):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                    img = img / 255.0
                    images.append(img)
                    labels.append(label)

    images = np.array(images).astype('float32')
    labels = to_categorical(np.array(labels), num_classes=len(class_names))

    indices = np.arange(len(images))
    np.random.shuffle(indices)
    images = images[indices]
    labels = labels[indices]

    print(f"[INFO] Classes detected: {class_names}")
    for i, class_name in enumerate(class_names):
        count = np.sum(np.argmax(labels, axis=1) == i)
        print(f"Class '{class_name}': {count} samples")

    return images, labels, class_names

# === Build MobileNetV2 Model ===
def build_model(num_classes, train_base=False):
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    base_model.trainable = train_base
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.5)(x)
    output = Dense(num_classes, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=output)
    loss_fn = tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1)
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.summary()
    return model

# === Load and Split Data ===
X, y, class_names = load_data()
y_indices = np.argmax(y, axis=1)
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_idx, test_idx in split.split(X, y_indices):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    y_train_indices = y_indices[train_idx]

class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_indices), y=y_train_indices)
class_weights_dict = dict(enumerate(class_weights))

# === Data Augmentation ===
data_gen = ImageDataGenerator(
    zoom_range=0.2,
    rotation_range=20,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.3, 1.8],
    shear_range=0.2,
    horizontal_flip=False,
    fill_mode='nearest'
)

# === Build Initial Model ===
model = build_model(len(class_names), train_base=False)

# === Callbacks ===
checkpoint_cb = ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1)

# === Training - Phase 1 (Frozen base model) ===
speak("Training in progress")
print("[INFO] Starting training...")
print(f"Time when Training Started --> {datetime.now().strftime('%H:%M:%S')}")

initial_epochs = 5
history = model.fit(
    data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=initial_epochs,
    callbacks=[checkpoint_cb, early_stop, reduce_lr],
    class_weight=class_weights_dict,
    shuffle=True
)

# === Unfreeze base model for fine-tuning ===
print("[INFO] Unfreezing base model for fine-tuning...")
model.trainable = True
model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
              metrics=['accuracy'])

fine_tune_epochs = EPOCHS - initial_epochs
history_fine = model.fit(
    data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
    validation_data=(X_test, y_test),
    epochs=fine_tune_epochs,
    callbacks=[checkpoint_cb, early_stop, reduce_lr],
    class_weight=class_weights_dict,
    shuffle=True
)

# === Evaluate ===
val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"[INFO] Final Validation Accuracy ---> {val_acc:.4f}")

# === Classification Report + Confusion Matrix ===
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

print("\n[INFO] Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("model/confusion_matrix.png")
print("[INFO] Confusion matrix saved to model/confusion_matrix.png")

# === Save Model ===
now = datetime.now().strftime("%H:%M:%S")
model.save(MODEL_H5_PATH)
model.save(MODEL_KERAS_PATH)
print(f"[INFO] Model saved at ----> {now}")
with open("model/labels.txt", "w") as f:
    for label in class_names:
        f.write(label + "\n")
print("[INFO] Labels saved to model/labels.txt")

# === Training Graphs ===
plt.figure(figsize=(14, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'] + history_fine.history['accuracy'], label='Train Accuracy', marker='o')
plt.plot(history.history['val_accuracy'] + history_fine.history['val_accuracy'], label='Val Accuracy', marker='o')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'] + history_fine.history['loss'], label='Train Loss', marker='o')
plt.plot(history.history['val_loss'] + history_fine.history['val_loss'], label='Val Loss', marker='o')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("model/training_metrics.png")
print("[INFO] Training metrics plot saved to model/training_metrics.png")

# === Done ===
final_accuracy = val_acc * 100
print(f"[INFO] Final Validation Accuracy ----> {final_accuracy:.2f}%")
play_beep()
speak("Training completed. All files saved to their respective directories")























# # --- TRAIN.PY ---  working code it is  -- was using this -- remember 
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import warnings
# warnings.filterwarnings("ignore", category=UserWarning, module="keras.src.trainers.data_adapters.py_dataset_adapter")

# import absl.logging  
# absl.logging.set_verbosity(absl.logging.FATAL)  

# import cv2
# # import warnings
# # import logging
# import numpy as np
# from datetime import datetime
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import ModelCheckpoint , EarlyStopping , ReduceLROnPlateau
# from tensorflow.keras.layers import BatchNormalization
# from sklearn.model_selection import train_test_split
# import tensorflow as tf
# import pyttsx3
# import winsound
# from tensorflow.keras.regularizers import l2    #type:ignore
# import matplotlib.pyplot as plt
# from tensorflow.keras.optimizers import Adam



# tts = pyttsx3.init()
# tts.setProperty('rate', 160)
# tts.setProperty('voice', tts.getProperty('voices')[1].id)

# def speak(text):
#     tts.say(text)
#     tts.runAndWait()

# def play_beep():
#     winsound.Beep(1000, 300)

# DATASET_PATH = 'dataset'
# MODEL_H5_PATH = 'model/model.h5'
# MODEL_KERAS_PATH = 'model/sign_model.keras'
# CHECKPOINT_PATH = 'checkpoint/model_checkpoint.keras'
# IMAGE_SIZE = 96
# EPOCHS = 30         #25
# BATCH_SIZE = 32

# def skin_mask(img):
#     # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     # lower = np.array([0, 48, 80], dtype=np.uint8)
#     # upper = np.array([20, 255, 255], dtype=np.uint8)
#     # skin = cv2.inRange(hsv, lower, upper)
#     # return cv2.bitwise_and(img, img, mask=skin)
    
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

#     # More specific range for skin tones (often works well)
#     lower_skin = np.array([0, 20, 40], dtype=np.uint8)
#     upper_skin = np.array([40, 255, 255], dtype=np.uint8)

#     # Additional range for reddish skin tones
#     lower_skin2 = np.array([140, 20, 40], dtype=np.uint8)
#     upper_skin2 = np.array([180, 255, 255], dtype=np.uint8)

#     # Create masks for both skin tone ranges
#     mask1 = cv2.inRange(hsv, lower_skin, upper_skin)
#     mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)

#     # Combine the masks
#     skin_mask = cv2.bitwise_or(mask1, mask2)
    
#     # Apply morphological operations to reduce noise
#     kernel = np.ones((3, 3), np.uint8)
#     skin_mask = cv2.erode(skin_mask, kernel, iterations=1)
#     skin_mask = cv2.dilate(skin_mask, kernel, iterations=1)

#     # Apply the mask to the original image
#     masked_img = cv2.bitwise_and(img, img, mask=skin_mask)

#     return masked_img
    
    

# def load_data():
#     print("[INFO] Loading dataset...")
#     images, labels, class_names = [], [], []

#     for folder_type in ['alphabet', 'word']:
#         folder_path = os.path.join(DATASET_PATH, folder_type)
#         if not os.path.exists(folder_path):
#             continue
#         # for class_name in sorted(os.listdir(folder_path)):
#         for class_name in sorted([d for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]):      #added
#             class_path = os.path.join(folder_path, class_name)
#             if os.path.isdir(class_path):
#                 if class_name not in class_names:
#                     class_names.append(class_name)
#                 label = class_names.index(class_name)
#                 for img_file in os.listdir(class_path):
#                     img_path = os.path.join(class_path, img_file)
#                     img = cv2.imread(img_path)
#                     if img is not None:
#                         # img = skin_mask(img)            #for skin masking       - was commented 
                        
#                         #these lines till break is added for now -----
#                         # cv2.imshow("Masked Image", img)
#                         # cv2.waitKey(500)  # Show each image for 0.5 seconds
#                         # if cv2.getWindowProperty('Masked Image', cv2.WND_PROP_VISIBLE) < 1:
#                         #     break # Break if window is closed
                        
                        
#                         img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))         #was using this only 
#                         img = img / 255.0               # Normalize each image here - added
#                         images.append(img)
#                         labels.append(label)

#     # images = np.array(images) / 255.0
#     # images = np.array(images).astype('float32') / 255.0         # was using these only
#     images = np.array(images).astype('float32')          #added
#     labels = to_categorical(np.array(labels), num_classes=len(class_names))
    
#     #added this --
#     # Shuffle the dataset
#     indices = np.arange(len(images))
#     np.random.shuffle(indices)
#     images = images[indices]
#     labels = labels[indices]

#     print(f"[INFO] Classes detected: {class_names}")
#     for i, class_name in enumerate(class_names):
#         sample_count = np.sum(np.argmax(labels, axis=1) == i)
#         print(f"Class '{class_name}': {sample_count} samples")
#     return images, labels, class_names


# # from tensorflow.keras.applications import MobileNetV2
# # from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
# # from tensorflow.keras.models import Model

# # def build_model(num_classes):
# #     base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))

# #     # Freeze the base model's layers
# #     for layer in base_model.layers:
# #         layer.trainable = False

# #     # Add a new classification head
# #     x = base_model.output
# #     x = GlobalAveragePooling2D()(x)
# #     x = Dense(256, activation='relu')(x)
# #     x = BatchNormalization()(x)
# #     x = Dropout(0.5)(x)
# #     predictions = Dense(num_classes, activation='softmax')(x)

# #     model = Model(inputs=base_model.input, outputs=predictions)

# #     model.compile(optimizer=Adam(learning_rate=0.0001),
# #                   loss='categorical_crossentropy',
# #                   metrics=['accuracy'])

# #     model.summary()
# #     return model


# def build_model(num_classes):
#     model = Sequential([
#         Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
#         Conv2D(64, (3, 3), activation='relu', padding='same'),                    #32
#         BatchNormalization(),
#         MaxPooling2D(2, 2),
     
#         Conv2D(128, (3, 3), activation='relu', padding='same'),          #64
#         BatchNormalization(),
#         MaxPooling2D(2, 2),
        
#         Conv2D(256, (3, 3), activation='relu', padding='same'),         #128
#         BatchNormalization(),
#         MaxPooling2D(2, 2),
        
#         Flatten(),
#         Dense(256, activation='relu', kernel_regularizer=l2(0.001)),        #128   --- was using this 
#         Dropout(0.5),             #was using this 
#         # Dropout(0.7),
#         Dense(num_classes, activation='softmax')
#     ])
#     # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
#     model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])  #added
#     model.summary()
#     return model



# X, y, class_names = load_data()
# # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)      using this
# from sklearn.model_selection import StratifiedShuffleSplit
# from sklearn.utils import class_weight

# # Convert one-hot encoded y to label indices
# y_indices = np.argmax(y, axis=1)

# # Stratified split: preserves class distribution
# split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
# for train_idx, test_idx in split.split(X, y_indices):
#     X_train, X_test = X[train_idx], X[test_idx]
#     y_train, y_test = y[train_idx], y[test_idx]
#     y_train_indices = y_indices[train_idx]      #added

# # Compute class weights             # added--- 
# # class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_indices), y=y_indices)
# class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(y_train_indices), y=y_train_indices)
# class_weights_dict = dict(enumerate(class_weights))


# data_gen = ImageDataGenerator(
#     # zoom_range=0.1,               #using this earlier
#     # width_shift_range=0.1,
#     # height_shift_range=0.1,
#     # brightness_range=[0.8, 1.2]
    
#     # zoom_range=0.2,
#     # rotation_range=20,
#     # width_shift_range=0.2,
#     # height_shift_range=0.2,
#     # brightness_range=[0.6, 1.4],
#     # shear_range=0.2,
#     # horizontal_flip=True
    
#     zoom_range=0.1,                 #was using this
#     rotation_range=10,              #was using this
#     width_shift_range=0.1,          #was using this
#     height_shift_range=0.1          #was using this
    
#     # zoom_range=0.2,
#     # rotation_range=25,
#     # width_shift_range=0.2,
#     # height_shift_range=0.2,
#     # brightness_range=[0.6, 1.4],
#     # shear_range=0.3,
#     # horizontal_flip=True,
#     # fill_mode='nearest'
    
    
#     # brightness_range=[0.6, 1.4],
#     # shear_range=0.2,
#     # horizontal_flip=True
    
#     # zoom_range=0.1,
#     # rotation_range=15,
#     # width_shift_range=0.1,
#     # height_shift_range=0.1,
#     # brightness_range=[0.8, 1.2],  # Wider brightness variation
#     # # shear_range=0.3,
#     # horizontal_flip=True,
#     # # vertical_flip=False,        # You might want to experiment with this
#     # fill_mode='nearest',        # How to fill in newly created pixels
#     # rescale=1./255
# )


# if os.path.exists(MODEL_H5_PATH):
#     model = load_model(MODEL_H5_PATH)
#     if model.output_shape[-1] != len(class_names):
#         print("[WARNING] Output mismatch. Rebuilding model.")
#         model = build_model(len(class_names))
#     else:
#         # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         model.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])      #added
# else:
#     model = build_model(len(class_names))

# checkpoint_cb = ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, monitor='val_accuracy', mode='max')
# early_stop = EarlyStopping(monitor='val_loss',  patience=5, restore_best_weights=True)       #added

# reduce_lr = ReduceLROnPlateau(
#     monitor='val_loss',
#     factor=0.5,
#     patience=3,
#     min_lr=1e-6,
#     verbose=1
# )



# speak("Training in progress")
# print("[INFO] Starting training...")
# now = datetime.now().strftime("%H:%M:%S")
# print(f"Time when Training Started --> {now}")

# # added --
# print("\n[DEBUG] Class Names Order:", class_names)
# print("[DEBUG] One-hot encoded shape:", y.shape)
# print("[DEBUG] First 10 training labels:", np.argmax(y_train[:10], axis=1))
# print("[DEBUG] First 10 validation labels:", np.argmax(y_test[:10], axis=1))



# history = model.fit(
#     data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
#     validation_data=(X_test, y_test),
#     epochs=EPOCHS,
#     callbacks=[checkpoint_cb, early_stop, reduce_lr],      #added
#     class_weight=class_weights_dict,  
#     shuffle=True                                           #added
#     # callbacks=[checkpoint_cb],
#     # callbacks=[checkpoint_cb, early_stop],               #added
# )



# # history = model.fit(
# #     data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
# #     validation_data=(X_test, y_test),
# #     epochs=EPOCHS,
# #     callbacks=[checkpoint_cb]
# #     class_weight=class_weights_dict
# # )
# # callbacks=[checkpoint_cb, early_stop]       #added

    


# # Evaluate model before saving      --  added
# val_loss, val_acc = model.evaluate(X_test, y_test, verbose=0)
# print(f"[INFO] Final Validation Accuracy from .evaluate() ---> {val_acc:.4f}")


# # Save model
# model.save(MODEL_H5_PATH)
# now = datetime.now().strftime("%H:%M:%S")
# print(f"[INFO] Model saved to ----> {MODEL_H5_PATH} at ----> {now}")

# model.save(MODEL_KERAS_PATH)
# now = datetime.now().strftime("%H:%M:%S")
# print(f"[INFO] Keras-native model saved to ----> {MODEL_KERAS_PATH} at ----> {now}")

# print(f"[INFO] Best checkpoint saved to ----> {CHECKPOINT_PATH}")

# # Save class labels
# with open("model/labels.txt", "w") as f:
#     for label in class_names:
#         f.write(label + "\n")
# print("[INFO] Labels saved to model/labels.txt")

# if os.path.exists(CHECKPOINT_PATH):
#     mod_time = os.path.getmtime(CHECKPOINT_PATH)
#     updated_at = datetime.fromtimestamp(mod_time).strftime("%H:%M:%S")
#     print(f"[INFO] Checkpoint updated at ----> {updated_at}")
# else:
#     print("[INFO] Checkpoint not found....!!")

# # Final report
# # final_accuracy = history.history['val_accuracy'][-1] * 100
# final_accuracy = val_acc * 100              #added
# print(f"[INFO] Final Validation Accuracy ----> {final_accuracy:.2f}%")
# print("[INFO] Training Completed Successfully.......")
# print(f"[INFO] Model training completed at ----> {datetime.now().strftime('%H:%M:%S')}")


# if not os.path.exists('model'):
#     os.makedirs('model')

# plt.figure(figsize=(14, 5))


# # Save training accuracy graph - added 
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'], label='Train Accuracy', marker='o')
# plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='o')
# plt.title('Model Accuracy Over Epochs')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy')
# # plt.title('Training vs Validation Accuracy')
# plt.legend()
# plt.grid(True)
# # plt.savefig("model/training_accuracy.png")
# # print("[INFO] Accuracy graph saved to model/training_accuracy.png")


# # Loss
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'], label='Training Loss', marker='o')
# plt.plot(history.history['val_loss'], label='Validation Loss', marker='o')
# # plt.title("Model Loss")
# plt.title('Model Loss Over Epochs')
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.grid(True)
# # plt.savefig('model/training_loss.png')
# # print("[INFO] Loss graph saved to model/training_loss.png")

# #  Add tight layout here before saving
# plt.tight_layout()

# # ✅ Save single combined plot
# plt.savefig("model/training_metrics.png")
# # plt.show()
# print("[INFO] Training accuracy/loss graph saved to model/training_metrics.png")


# # ✅ Completion (beep + TTS)
# play_beep()
# speak("Training completed. All files saved to their respective directories")

































































# # train.py - this also works
# import os
# # ✅ Suppress logs and warnings
# # -------------------------
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import absl.logging
# import warnings
# # -------------------------
# absl.logging.set_verbosity(absl.logging.FATAL)
# warnings.filterwarnings("ignore", category=UserWarning, module='keras')



# import sys
# import logging
# import pyttsx3
# import winsound
# import cv2
# import numpy as np
# from datetime import datetime
# from tensorflow.keras.models import Sequential, load_model
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.preprocessing.image import ImageDataGenerator
# from tensorflow.keras.callbacks import ModelCheckpoint
# from sklearn.model_selection import train_test_split
# import tensorflow as tf


# # -------------------------
# # ✅ TTS setup
# # -------------------------
# tts = pyttsx3.init()
# tts.setProperty('rate', 160)
# voices = tts.getProperty('voices')
# tts.setProperty('voice', voices[1].id)

# def speak(text):
#     tts.say(text)
#     tts.runAndWait()

# def play_beep():
#     winsound.Beep(1000, 300)

# # -------------------------
# # ✅ Config
# # -------------------------
# DATASET_PATH = 'dataset'
# MODEL_H5_PATH = 'model/model.h5'
# MODEL_KERAS_PATH = 'model/sign_model.keras'
# CHECKPOINT_PATH = 'checkpoint/model_checkpoint.keras'
# IMAGE_SIZE = 96
# EPOCHS = 25
# BATCH_SIZE = 32

# # -------------------------
# # ✅ Load data with imbalance check
# # -------------------------
# def load_data():
#     print("[INFO] Loading dataset...")
#     images, labels, class_names = [], [], []

#     for folder_type in ['alphabet', 'word']:
#         folder_path = os.path.join(DATASET_PATH, folder_type)
#         if not os.path.exists(folder_path):
#             continue
#         for class_name in sorted(os.listdir(folder_path)):
#             class_path = os.path.join(folder_path, class_name)
#             if os.path.isdir(class_path):
#                 if class_name not in class_names:
#                     class_names.append(class_name)
#                 label = class_names.index(class_name)
#                 for img_file in os.listdir(class_path):
#                     img_path = os.path.join(class_path, img_file)
#                     img = cv2.imread(img_path)
#                     if img is not None:
#                         img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#                         images.append(img)
#                         labels.append(label)

#     images = np.array(images) / 255.0
#     labels = to_categorical(np.array(labels), num_classes=len(class_names))

#     print(f"[INFO] Classes detected: {class_names}")

#     # ✅ Imbalance check
#     class_counts = {name: 0 for name in class_names}
#     for lbl in np.argmax(labels, axis=1):
#         class_counts[class_names[lbl]] += 1

#     print("[INFO] Class distribution:")
#     for name, count in class_counts.items():
#         print(f"  {name}: {count} images")

#     if max(class_counts.values()) / max(1, min(class_counts.values())) > 1.5:
#         print("[⚠️ WARNING] Dataset is imbalanced. Consider adding more samples to underrepresented classes.")

#     return images, labels, class_names

# # -------------------------
# # ✅ Build model
# # -------------------------
# def build_model(num_classes):
#     model = Sequential([
#         Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
#         Conv2D(32, (3, 3), activation='relu'),
#         MaxPooling2D(2, 2),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D(2, 2),
#         Conv2D(128, (3, 3), activation='relu'),
#         MaxPooling2D(2, 2),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     model.summary()
#     return model

# # -------------------------
# # ✅ Train
# # -------------------------
# X, y, class_names = load_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# data_gen = ImageDataGenerator(
#     zoom_range=0.1,
#     width_shift_range=0.1,
#     height_shift_range=0.1,
#     brightness_range=[0.8, 1.2]
# )

# rebuild_model = False
# unique_classes = len(class_names)

# if os.path.exists(MODEL_H5_PATH):
#     print("[INFO] Found existing model. Loading...")
#     model = load_model(MODEL_H5_PATH)
#     if model.output_shape[-1] != unique_classes:
#         print("[WARNING] Output class mismatch. Rebuilding model...")
#         rebuild_model = True
#     else:
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# else:
#     rebuild_model = True

# if rebuild_model:
#     model = build_model(unique_classes)

# # Checkpoint
# checkpoint_cb = ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, monitor='val_accuracy', mode='max')

# # Start training
# speak("Training in progress")
# print("[INFO] Starting training...")

# history = model.fit(
#     data_gen.flow(X_train, y_train, batch_size=BATCH_SIZE),
#     validation_data=(X_test, y_test),
#     epochs=EPOCHS,
#     callbacks=[checkpoint_cb]
# )

# # Save model
# model.save(MODEL_H5_PATH)
# model.save(MODEL_KERAS_PATH)
# now = datetime.now().strftime("%H:%M:%S")
# print(f"[INFO] Model saved at {now}")
# print(f"[INFO] Best checkpoint saved to ----> {CHECKPOINT_PATH}")

# # Save label list
# with open("model/labels.txt", "w") as f:
#     for label in class_names:
#         f.write(label + "\n")
# print("[INFO] Labels saved to model/labels.txt")

# # Final status
# final_accuracy = history.history['val_accuracy'][-1] * 100
# print(f"[INFO] Final Validation Accuracy ----> {final_accuracy:.2f}%")
# print("[INFO] Training Completed Successfully.......")
# print(f"[INFO] Model training completed at ----> {datetime.now().strftime('%H:%M:%S')}")

# play_beep()
# speak("Training completed. All files saved to their respective directories")






















# correct working code --- was using this earlier ( good one )
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# import absl.logging
# absl.logging.set_verbosity(absl.logging.ERROR)

# import tensorflow as tf
# import pyttsx3           # ✅ TTS
# import winsound          # ✅ Beep (Windows only)
# import pickle
# import cv2
# import numpy as np
# from datetime import datetime
# from tensorflow.keras.models import Sequential, load_model      #type:ignore
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import ModelCheckpoint

# # ✅ Text-to-Speech Setup
# tts = pyttsx3.init()
# tts.setProperty('rate', 160)
# voices = tts.getProperty('voices')
# tts.setProperty('voice', voices[1].id)  # Change index as needed

# def speak(text):
#     tts.say(text)
#     tts.runAndWait()

# def play_beep():
#     winsound.Beep(1000, 300)  # frequency, duration (ms)

# # Constants
# DATASET_PATH = 'dataset'
# MODEL_H5_PATH = 'model/model.h5'
# MODEL_KERAS_PATH = 'model/sign_model.keras'
# CHECKPOINT_PATH = 'checkpoint/model_checkpoint.keras'
# IMAGE_SIZE = 64
# EPOCHS = 10
# BATCH_SIZE = 32

# def load_data():
#     print("[INFO] Loading dataset...")
#     images = []
#     labels = []
#     class_names = []

#     for folder_type in ['alphabet', 'word']:
#         folder_path = os.path.join(DATASET_PATH, folder_type)
#         if not os.path.exists(folder_path):
#             continue
#         for class_name in os.listdir(folder_path):
#             class_path = os.path.join(folder_path, class_name)
#             if not os.path.isdir(class_path):
#                 continue
#             if class_name not in class_names:
#                 class_names.append(class_name)
#             label = class_names.index(class_name)
#             for img_file in os.listdir(class_path):
#                 img_path = os.path.join(class_path, img_file)
#                 img = cv2.imread(img_path)
#                 if img is None:
#                     continue
#                 img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#                 images.append(img)
#                 labels.append(label)

#     images = np.array(images) / 255.0
#     labels = to_categorical(np.array(labels), num_classes=len(class_names))
#     print(f"[INFO] Classes detected: {class_names}")
#     return images, labels, class_names

# def build_model(num_classes):
#     model = Sequential([
#         Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
#         Conv2D(32, (3, 3), activation='relu'),
#         MaxPooling2D(2, 2),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D(2, 2),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # Load data
# X, y, class_names = load_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Load or build model
# rebuild_model = False
# unique_classes = len(class_names)

# if os.path.exists(MODEL_H5_PATH):
#     print("[INFO] Found existing model. Loading...")
#     model = load_model(MODEL_H5_PATH)
#     existing_output_classes = model.output_shape[-1]
#     if existing_output_classes != unique_classes:
#         print(f"[WARNING] Existing model has {existing_output_classes} output classes but dataset has {unique_classes}.")
#         print("[INFO] Rebuilding model to match new class count.")
#         rebuild_model = True
#     else:
#         print("[INFO] Model loaded successfully. Compiling...")
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         model.evaluate(X_test[:1], y_test[:1], verbose=0)
# else:
#     rebuild_model = True

# if rebuild_model:
#     model = build_model(unique_classes)

# # Checkpoint
# checkpoint_cb = ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, monitor='val_accuracy', mode='max')

# # ✅ Start Training (voice)
# speak("Training in progress")

# print("[INFO] Starting training...")
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_test, y_test),
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     callbacks=[checkpoint_cb]
# )

# # Save model
# model.save(MODEL_H5_PATH)
# now = datetime.now().strftime("%H:%M:%S")
# print(f"[INFO] Model saved to ----> {MODEL_H5_PATH} at ----> {now}")

# model.save(MODEL_KERAS_PATH)
# now = datetime.now().strftime("%H:%M:%S")
# print(f"[INFO] Keras-native model saved to ----> {MODEL_KERAS_PATH} at ----> {now}")

# print(f"[INFO] Best checkpoint saved to ----> {CHECKPOINT_PATH}")

# # Save class labels
# with open("model/labels.txt", "w") as f:
#     for label in class_names:
#         f.write(label + "\n")
# print("[INFO] Labels saved to model/labels.txt")

# if os.path.exists(CHECKPOINT_PATH):
#     mod_time = os.path.getmtime(CHECKPOINT_PATH)
#     updated_at = datetime.fromtimestamp(mod_time).strftime("%H:%M:%S")
#     print(f"[INFO] Checkpoint updated at ----> {updated_at}")
# else:
#     print("[INFO] Checkpoint not found....!!")

# # Final report
# final_accuracy = history.history['val_accuracy'][-1] * 100
# print(f"[INFO] Final Validation Accuracy ----> {final_accuracy:.2f}%")
# print("[INFO] Training Completed Successfully.......")
# print(f"[INFO] Model training completed at ----> {datetime.now().strftime('%H:%M:%S')}")

# # ✅ Completion (beep + TTS)
# play_beep()
# speak("Training completed. All files saved to their respective directories")





































# without sound
# import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Suppress TensorFlow INFO and WARNING logs - just to bypass warnings
# os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN for consistent numerical ops - just to bypass warnings

# import absl.logging
# absl.logging.set_verbosity(absl.logging.ERROR)

# import tensorflow as tf
# tf.get_logger().setLevel('ERROR')

# import pickle 

# import cv2
# import numpy as np
# from datetime import datetime
# from tensorflow.keras.models import Sequential, load_model  
# from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout
# from tensorflow.keras.utils import to_categorical
# from sklearn.model_selection import train_test_split
# from tensorflow.keras.callbacks import ModelCheckpoint

# DATASET_PATH = 'dataset'
# MODEL_H5_PATH = 'model/model.h5'
# MODEL_KERAS_PATH = 'model/sign_model.keras'
# CHECKPOINT_PATH = 'checkpoint/model_checkpoint.keras'
# IMAGE_SIZE = 64
# EPOCHS = 10
# BATCH_SIZE = 32

# def load_data():
#     print("[INFO] Loading dataset...")
#     images = []
#     labels = []
#     class_names = []

#     for folder_type in ['alphabet', 'word']:
#         folder_path = os.path.join(DATASET_PATH, folder_type)
#         if not os.path.exists(folder_path):
#             continue
#         for class_name in os.listdir(folder_path):
#             class_path = os.path.join(folder_path, class_name)
#             if not os.path.isdir(class_path):
#                 continue
#             if class_name not in class_names:
#                 class_names.append(class_name)
#             label = class_names.index(class_name)
#             for img_file in os.listdir(class_path):
#                 img_path = os.path.join(class_path, img_file)
#                 img = cv2.imread(img_path)
#                 if img is None:
#                     continue
#                 img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
#                 images.append(img)
#                 labels.append(label)

#     images = np.array(images) / 255.0
#     labels = to_categorical(np.array(labels), num_classes=len(class_names))
#     print(f"[INFO] Classes detected: {class_names}")
#     return images, labels, class_names

# def build_model(num_classes):
#     model = Sequential([
#         Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
#         Conv2D(32, (3, 3), activation='relu'),
#         MaxPooling2D(2, 2),
#         Conv2D(64, (3, 3), activation='relu'),
#         MaxPooling2D(2, 2),
#         Flatten(),
#         Dense(128, activation='relu'),
#         Dropout(0.5),
#         Dense(num_classes, activation='softmax')
#     ])
#     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#     return model

# # def build_model(num_classes):
# #     model = Sequential([
# #         Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)),
# #         MaxPooling2D(2, 2),
# #         Conv2D(64, (3, 3), activation='relu'),
# #         MaxPooling2D(2, 2),
# #         Flatten(),
# #         Dense(128, activation='relu'),
# #         Dropout(0.5),
# #         Dense(num_classes, activation='softmax')
# #     ])
# #     model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# #     return model





# # Load data --
# X, y, class_names = load_data()
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# # Load or build model -- ( advance technique - if mismatched, it will be rebuild)
# rebuild_model = False
# unique_classes = len(class_names)

# if os.path.exists(MODEL_H5_PATH):
#     print("[INFO] Found existing model. Loading...")
#     model = load_model(MODEL_H5_PATH)
#     existing_output_classes = model.output_shape[-1]
#     if existing_output_classes != unique_classes:
#         print(f"[WARNING] Existing model has {existing_output_classes} output classes but dataset has {unique_classes}.")
#         print("[INFO] Rebuilding model to match new class count.")
#         rebuild_model = True
#     else:
#         print("[INFO] Model loaded successfully. Compiling...")
#         model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#         model.evaluate(X_test[:1], y_test[:1], verbose=0)  # Fix: Compile metrics
# else:
#     rebuild_model = True

# if rebuild_model:
#     model = build_model(unique_classes)

# # Setup checkpoint
# checkpoint_cb = ModelCheckpoint(CHECKPOINT_PATH, save_best_only=True, monitor='val_accuracy', mode='max')

# # Train
# print("[INFO] Starting training...")
# history = model.fit(
#     X_train, y_train,
#     validation_data=(X_test, y_test),
#     epochs=EPOCHS,
#     batch_size=BATCH_SIZE,
#     callbacks=[checkpoint_cb]
# )

# # Save model in both formats with timestamps --- 
# model.save(MODEL_H5_PATH)
# now = datetime.now().strftime("%H:%M:%S")
# print(f"[INFO] Model saved to ----> {MODEL_H5_PATH} at ----> {now}")

# model.save(MODEL_KERAS_PATH)
# now = datetime.now().strftime("%H:%M:%S")
# print(f"[INFO] Keras-native model saved to ----> {MODEL_KERAS_PATH} at ----> {now}")

# print(f"[INFO] Best checkpoint saved to ----> {CHECKPOINT_PATH}")


# # ✅ Save class names to labels.txt
# with open("model/labels.txt", "w") as f:
#     for label in class_names:
#         f.write(label + "\n")
# print("[INFO] Labels saved to model/labels.txt")



# if os.path.exists(CHECKPOINT_PATH):
#     mod_time = os.path.getmtime(CHECKPOINT_PATH)
#     updated_at = datetime.fromtimestamp(mod_time).strftime("%H:%M:%S")
#     print(f"[INFO] Checkpoint updated at ----> {updated_at}")
# else:
#     print("[INFO] Checkpoint not found....!!")


# # Final report
# final_accuracy = history.history['val_accuracy'][-1] * 100
# print(f"[INFO] Final Validation Accuracy ----> {final_accuracy:.2f}%")
# print("[INFO] Training Completed Successfully.......")
# now = datetime.now().strftime("%H:%M:%S")
# print(f"[INFO] Model training completed at ----> {now}")



