# Dogs vs. Muffins Classifier 🐶🧁

This project is an image classification web application developed for the **Semester 3 - Data Dynamics** module. It addresses the challenge of distinguishing between visually similar images—specifically Chihuahuas and Muffins—using Deep Learning.

## 🚀 Features
- **High-Accuracy Classification:** Built with a Convolutional Neural Network (CNN) using **TensorFlow** and **Keras**.
- **Explainable AI (XAI):** Integrated **Grad-CAM** (Gradient-weighted Class Activation Mapping) to provide transparency by visualizing which image features (e.g., eyes, paws, texture) influenced the model's decision.
- **Web Interface:** A user-friendly **Flask** application that allows for real-time image uploads and instant inference.
- **CI/CD Integration:** Automated workflows via GitHub Actions to ensure continuous code quality.

## 🛠️ Tech Stack
- **Languages:** Python (Main Project), R (Biweekly Tasks)
- **Deep Learning:** TensorFlow, Keras
- **Web Framework:** Flask
- **Computer Vision:** OpenCV (for Grad-CAM processing)
- **Deployment/QA:** GitHub Actions (CI/CD)

## 📦 Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Diogo-Alves04/DogsVSmuffins.git](https://github.com/Diogo-Alves04/DogsVSmuffins.git)
   cd DogsVSmuffins
Create and activate a virtual environment:
python -m venv venv
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate


Install dependencies:
pip install -r requirements.txt


Run the application:
python app.py
Access the interface at http://127.0.0.1:5000

🔍 Responsible AI & Transparency
A core pillar of this semester was the application of Responsible Data Science. This project ensures transparency by providing visual justifications for every prediction. By using Grad-CAM heatmaps, we can verify that the model is identifying biological features (like eyes or snout) rather than being misled by background textures or muffin liners.

🎥 Demonstration
A full walkthrough of the project, including the technical implementation and the live Grad-CAM demonstration, can be found here:

👉 Watch the Project Demo on YouTube

👤 Author
Diogo Alves - Semester 3 (Data Dynamics)