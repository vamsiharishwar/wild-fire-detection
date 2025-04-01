from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

# Load Trained Model
model = tf.keras.models.load_model("fire_detection_vgg16_finetuned.keras")

# Define Class Labels
class_names = ["Fire", "Non-Fire"]

# Initialize Flask App
app = Flask(__name__)

# Define Upload Folder
UPLOAD_FOLDER = "static/uploads"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Function to Predict Fire in Image
def predict_fire(image_path):
    IMG_SIZE = (224, 224)  # Resize to match model input size
    image = cv2.imread(image_path)  # Read image
    image = cv2.resize(image, IMG_SIZE)  # Resize
    image = image / 255.0  # Normalize
    image = np.expand_dims(image, axis=0)  # Expand dimensions for model input

    prediction = model.predict(image)
    predicted_class = np.argmax(prediction)  # Get class with highest probability
    confidence = np.max(prediction)  # Confidence score

    return class_names[predicted_class], confidence

# Flask Routes
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        if "file" not in request.files:
            return render_template("index.html", message="No file uploaded!")

        file = request.files["file"]
        if file.filename == "":
            return render_template("index.html", message="No selected file!")

        if file:
            filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
            file.save(filepath)  # Save uploaded file

            # Get Prediction
            prediction, confidence = predict_fire(filepath)

            return render_template("index.html", uploaded_image=filepath, prediction=prediction, confidence=confidence)

    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
