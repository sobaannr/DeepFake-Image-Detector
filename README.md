# Deepfake Image Detection Project

## Overview
This project is a **Deepfake Image Detector** using **Convolutional Neural Networks (CNNs)**. It detects whether an image is a **real face** or a **deepfake**. The project workflow includes:

- Data preprocessing and visualization in **Google Colab**
- Training a CNN model in Colab
- Saving the trained model
- Using the model for inference in **VS Code** or a web interface (Flask)

---

## Dataset

The dataset used is the **Deepfake and Real Images** dataset from Kaggle:

- [Dataset URL](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)

- Structure:
Dataset/
Train/
Real/
Fake/
Validation/
Real/
Fake/
Test/
Real/
Fake/

- Each class (`Real` and `Fake`) contains images resized to 128x128 during preprocessing.

> **Important:** In Colab, the dataset was downloaded directly via Kaggle API.

---

## Google Colab Notebook Workflow

### 1. Dataset Download
- Installed Kaggle API and authenticated using `kaggle.json`
- Downloaded dataset directly into Colab

### 2. Data Preprocessing
- Resized images to `128x128` using **OpenCV**  
- Normalized pixel values to `[0,1]`  
- Converted labels into numerical format: `Real=0`, `Fake=1`

### 3. Data Visualization
- Bar charts to show the number of images per class
- Sample image grids to visualize dataset distribution

### 4. Train/Test Split
- Split dataset into train/test sets using `train_test_split` (80/20)

### 5. CNN Model
- Built a sequential CNN with `Conv2D`, `MaxPooling2D`, `Flatten`, `Dense`, and `Dropout`
- Used `sparse_categorical_crossentropy` as the loss function
- Trained for 10 epochs (batch size 32)

### 6. Training Visualization
- Plotted training/validation accuracy and loss graphs
- Saved the trained model as `deepfake_detector.h5` in `/content/models/` folder

> **Note:** The Colab runtime is temporary. To preserve the model, it was saved to Google Drive or downloaded immediately.

---

## VS Code Setup

### Folder Structure
deepfake-project/
├── models/
│ └── deepfake_detector.h5
├── utils/
│ ├── preprocess.py # Optional preprocessing functions
│ └── predict.py # Load model and predict
├── app.py # Flask app for manual upload and prediction
├── static/ # Uploads handled by Flask
├── templates/ # HTML templates for Flask
├── README.md


### How to Run

#### 1. Test Predictions in VS Code
- Load model from `models/deepfake_detector.h5`:

```python
from tensorflow.keras.models import load_model
from utils.predict import preprocess_image, predict_image

# Load trained model
model = load_model("models/deepfake_detector.h5")

# Predict on a new image
img_path = "static/test_image.jpg"
result = predict_image(model, img_path)
print("Prediction:", result)  # Output: "Real" or "Fake"

2. Flask App 

Run the Flask app:

Upload any image and get real-time prediction

Libraries Used
Python 3.x

TensorFlow / Keras

NumPy

OpenCV (for reading and resizing images)

Matplotlib (data visualization)

scikit-learn (train/test split)

Flask (optional for web interface)

Demo Video
A demo video showing the workflow and predictions will be posted on LinkedIn soon.

Author
Sobaan Nayyar

GitHub: https://github.com/sobaannr

Email: sobaannr@gmail.com

References
Kaggle Dataset: Deepfake and Real Images