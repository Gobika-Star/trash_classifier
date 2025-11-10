# scripts/predict_image.py
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array # type: ignore
import numpy as np
import os

MODEL_PATH = "model.h5"
IMAGE_PATH = input("Enter image path (e.g., test.jpg): ")

if not os.path.exists(MODEL_PATH):
    print(f"❌ Model not found at {MODEL_PATH}")
    exit()

if not os.path.exists(IMAGE_PATH):
    print(f"❌ Image not found at {IMAGE_PATH}")
    exit()

model = tf.keras.models.load_model(MODEL_PATH)

# Load and preprocess image
img = load_img(IMAGE_PATH, target_size=(224,224))
x = img_to_array(img) / 255.0
x = np.expand_dims(x, axis=0)

# Predict
preds = model.predict(x)
class_index = np.argmax(preds)
class_names = ["Plastic", "Paper", "Metal", "Glass", "Organic", "Other"]
print(f"✅ Predicted class: {class_names[class_index]}")
