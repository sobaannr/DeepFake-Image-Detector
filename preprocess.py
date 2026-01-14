import cv2
import numpy as np

IMG_SIZE = 128

def preprocess_image(img_path):
    
    """
    Read an image from img_path, resize, normalize, and return as array.
    """
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Image not found: {img_path}")
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    img = img / 255.0
    
    img = np.expand_dims(img, axis=0)
    return img
