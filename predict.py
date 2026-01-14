import tensorflow as tf
from tensorflow.keras.models import load_model
from .preprocess import preprocess_image
import numpy as np


MODEL_PATH = "model/deepfake_detector.h5"
model = load_model(MODEL_PATH)

def predict_image(img_path):
    
    """
    Predict if an image is Real or Fake.
    Returns "Real" or "Fake" and probability scores.
    """
    img_array = preprocess_image(img_path)
    pred = model.predict(img_array)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx]
    label = "Fake" if class_idx == 1 else "Real"
    return label, confidence
