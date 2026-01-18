from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
import os
from utils.predict import predict_image

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = ""
    confidence = 0.0
    image_path = ""  
    
    if request.method == "POST":
        if 'file' not in request.files:
            return "No file part"
        file = request.files['file']
        if file.filename == '':
            return "No selected file"
        if file:
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            image_path = f"uploads/{filename}"  
            prediction, confidence = predict_image(filepath)

    return render_template("index.html", prediction=prediction, confidence=confidence, image_path=image_path)  # Modified this line

if __name__ == "__main__":
    app.run(debug=True)