import cv2
from keras.models import load_model
import numpy as np

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load the emotion classification model
emotion_model = load_model('smile_detection_model_bigData.h5')

# Read the input image
image_path = "C:/Users/jania/OneDrive/Desktop/MEGHA/MEGHA_projects/My datasets/my face/sad (4).jpg"
image = cv2.imread(image_path)

# Convert the image to grayscale (face detection works on grayscale images)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the grayscale image
faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

# Loop over each detected face
for (x, y, w, h) in faces:
    # Extract the region of interest (ROI) corresponding to the detected face
    face_roi = image[y:y + h, x:x + w]

    # Resize the face ROI to match model input size (e.g., 64x64 pixels)
    face_roi_resized = cv2.resize(face_roi, (64, 64))
    # face_roi_resized = face_roi_resized/255.0
    # Perform any additional preprocessing (e.g., normalization) if required
    # face_roi_processed = preprocess_input(face_roi_resized)

    # Predict emotion on the face ROI
    emotion_prediction = emotion_model.predict(face_roi_resized[np.newaxis, ...])

    # Display the prediction (you can modify this part as needed)
    is_smiling = emotion_prediction[0][0] > 0.5
    if is_smiling:
        print("Smile detected!")
    else:
        print("No smile detected.")

    # Display prediction on frame
    text = "Smiling, Prediction: " + str(emotion_prediction[0][0]) if is_smiling else "Not Smiling, Prediction: " + str(emotion_prediction[0][0])
    cv2.putText(image, text, (10, 30), cv2.QT_FONT_NORMAL, 1, (255, 255, 255), 2)
    cv2.rectangle(image, (x, y), (x + w, y + h), (255, 255, 255), 2)


# Display the input image with detected faces
cv2.imshow('Detected Faces', image)
cv2.waitKey(0)
cv2.destroyAllWindows()
