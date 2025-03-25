import cv2
import numpy as np
from keras.models import load_model

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('C:/Users/jania/OneDrive/Desktop/MEGHA/MEGHA_projects/'
                                     'My datasets/haarcascade_frontalface_default.xml')

# Load the emotion classification model
emotion_model = load_model('smile_detection_model_bigData.h5')

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame1 = cap.read()

    # Convert the image to grayscale (face detection works on grayscale images)
    gray_image = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale image
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Loop over each detected face
    for (x, y, w, h) in faces:
        # Extract the region of interest (ROI) corresponding to the detected face
        face_roi = frame1[y:y + h, x:x + w]

        # Resize the face ROI to match model input size (e.g., 64x64 pixels)
        face_roi_resized = cv2.resize(face_roi, (64, 64))

        # frame = frame / 255.0  # Normalize pixel values to range [0, 1]

        # Predict emotion
        emotion_prediction = emotion_model.predict(face_roi_resized[np.newaxis, ...])
        is_smiling = (emotion_prediction[0][0] > 0.5)  # Assuming binary classification

        # Display prediction on frame
        text = "Smiling, Prediction: " + str(emotion_prediction[0][0]) if is_smiling else "Not Smiling, Prediction: " + str(emotion_prediction[0][0])
        cv2.putText(frame1, text, (x, y-10), cv2.QT_FONT_NORMAL, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.rectangle(frame1, (x, y), (x + w, y + h), (255, 255, 255), 2)

    # Resize the frame to the desired width and height
    resized_frame = cv2.resize(frame1, (800, 600))

    # Display frame
    cv2.imshow('Smile Detection', resized_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
