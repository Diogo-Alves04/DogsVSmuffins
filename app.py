import os
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing import image
from gradcam import make_gradcam_heatmap, overlay_heatmap

# Create the Flask application
app = Flask(__name__)

# Folder where uploaded images will be stored
UPLOAD_FOLDER = "static/uploads"

# Folder where Grad-CAM heatmap images will be stored
HEATMAP_FOLDER = "static/heatmaps"

# Create the folders if they do not already exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(HEATMAP_FOLDER, exist_ok=True)

# Load the trained CNN model for Dogs vs Muffins classification
model = tf.keras.models.load_model("model/dogs_vs_muffins_cnn.h5")

# Define the main route of the application
# Supports both GET (page load) and POST (image upload)
@app.route("/", methods=["GET", "POST"])
def index():
    # Initialize variables to avoid errors before prediction
    prediction = None
    confidence = None
    img_path = None
    heatmap_path = None

    # Check if the request method is POST (image submitted)
    if request.method == "POST":
        # Get the uploaded image from the form
        file = request.files["image"]

        # Define the path where the image will be saved
        img_path = os.path.join(UPLOAD_FOLDER, file.filename)

        # Save the uploaded image to disk
        file.save(img_path)

        # Load the image and resize it to match the model input size
        img = image.load_img(img_path, target_size=(150, 150))

        # Convert the image to a NumPy array and normalize pixel values
        img_array = image.img_to_array(img) / 255.0

        # Add a batch dimension (required by the model)
        img_array = np.expand_dims(img_array, axis=0)

        # -------- PREDICTION --------
        # Make a prediction using the trained model
        # Output is a probability value between 0 and 1
        pred = model.predict(img_array)[0][0]

        # If probability is greater than 0.5, classify as Muffin
        if pred > 0.5:
            prediction = "Muffin"
            confidence = round(pred * 100, 2)
        # Otherwise, classify as Dog
        else:
            prediction = "Dog"
            confidence = round((1 - pred) * 100, 2)

        # -------- GRAD-CAM --------
        # Generate the Grad-CAM heatmap to visualize model attention
        heatmap = make_gradcam_heatmap(img_array, model)

        # Define the path where the heatmap image will be saved
        heatmap_path = os.path.join(HEATMAP_FOLDER, file.filename)

        # Overlay the heatmap on the original image and save it
        overlay_heatmap(img_path, heatmap, heatmap_path)

    # Render the HTML template and pass prediction results
    return render_template(
        "index.html",
        prediction=prediction,
        confidence=confidence,
        img_path=img_path,
        heatmap_path=heatmap_path
    )

# Run the Flask app in debug mode
if __name__ == "__main__":
    app.run(debug=True)
