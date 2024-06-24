# maen.py

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('final_ensemble_emotion_model.keras')

# Load the face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels (adjust these based on your training labels)
emotion_labels = ['happy', 'sad']  # Replace with your actual labels

# Function to preprocess the face for prediction
def preprocess_face(face):
    face = cv2.resize(face, (48, 48))
    face = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB) # Convert to 3 channels
    face = face / 255.0
    face = np.expand_dims(face, axis=0)
    return [face, face]

# Open the camera
cap = cv2.VideoCapture(0)

plt.ion()  # Enable interactive mode
fig, ax = plt.subplots()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw rectangle around the faces and predict emotion
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        preprocessed_face = preprocess_face(face)
        predictions = model.predict(preprocessed_face)
        predicted_label = np.argmax(predictions)
        emotion = emotion_labels[predicted_label]

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Display the resulting frame using matplotlib
    ax.clear()
    ax.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    plt.pause(0.001)

    # Break the loop on 'q' key press
    if plt.waitforbuttonpress(0.001) and plt.get_current_fig_manager().canvas.manager.keypress_handler_id:
        break

# When everything is done, release the capture
cap.release()
plt.ioff()
plt.show()
