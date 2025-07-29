import cv2
import numpy as np
from tensorflow.keras.models import load_model          #type:ignore

model = load_model("model/model.h5")

with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

img = cv2.imread("sample.jpg")
img = cv2.resize(img, (64, 64))
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = img / 255.0
img = np.expand_dims(img, axis=0)

pred = model.predict(img)
conf = np.max(pred)
label = labels[np.argmax(pred)]

print("Predicted:", label)
print("Confidence:", conf)


