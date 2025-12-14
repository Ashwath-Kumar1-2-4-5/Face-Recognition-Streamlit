📸 Face Recognition Web App (Streamlit)

Live Demo: https://findyourface.streamlit.app

A web-based face recognition application built with Python and Streamlit. Users can upload images or use a webcam to detect and recognize faces using a trained model (Keras / custom dataset). Ideal for learning computer vision, real-time detection, and deploying ML apps quickly with Streamlit.

🧠 Features

✔️ Detect and recognize faces in images
✔ Real-time webcam face detection & recognition
✔ Custom trained face model included (face_model.keras)
✔ Name mapping for recognized identities (name_mapping.json)
✔ Simple and interactive web interface with Streamlit
✔ Deployed live using Streamlit Cloud

🚀 Demo

Access the live deployed version here:

👉 https://findyourface.streamlit.app

Share with others or integrate it into your portfolio!

🗂️ Repository Structure
📦Face-Recognition-Streamlit
 ┣ 📜.gitignore
 ┣ 📜app.py
 ┣ 📜train_model.py
 ┣ 📜requirements.txt
 ┣ 📜class_names.json
 ┣ 📜name_mapping.json
 ┗ 📜face_model.keras


app.py – Main Streamlit app for face detection & recognition

train_model.py – Script to train/generate face model

face_model.keras – Pre-trained Keras face classification model

class_names.json – List of class labels

name_mapping.json – Maps model output to human-readable names

requirements.txt – Python dependencies

🧩 Installation (Local)

⚙️ Make sure you have Python 3.8+ installed.

Clone the repository

git clone https://github.com/Ashwath-Kumar1-2-4-5/Face-Recognition-Streamlit.git
cd Face-Recognition-Streamlit


Create & activate a virtual environment (optional but recommended)

python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate


Install dependencies

pip install -r requirements.txt


Run the app

streamlit run app.py


🎉 The app should now be live at http://localhost:8501.

🧠 How It Works

Face Detection:

The app accepts an image upload or webcam feed.

Detects faces using computer vision or deep learning.

Face Recognition:

Cropped faces are passed through a Keras model (face_model.keras) trained to classify known identities.

Recognized face names are shown on screen with bounding boxes.

📌 The model and name mappings help convert raw predictions into readable labels.

📁 Training Your Own Model

If you want to train the face recognition model on your own dataset:

Prepare a dataset with folders for each person (e.g., dataset/Ashwaaa/, dataset/John/).

Modify and run:

python train_model.py


Replace the generated face_model.keras and update name_mapping.json.

📦 Requirements

These packages (from requirements.txt) are essential:

streamlit

tensorflow / keras

opencv-python

numpy

pillow

(Install via pip install -r requirements.txt)

📌 Notes

Works best with good lighting and frontal faces.

For webcam features, allow camera access when prompted. 
face-recognition-application.streamlit.app

👍 Contributing

Contributions are welcome!
Feel free to open issues or submit pull requests to improve features, detection logic, UI/UX, or performance.

📫 Contact

Built by Ashwath Kumar — feel free to reach out if you need help, demo ideas, or improvements.

⭐ If this project helped you, don’t forget to ⭐ the repo!
