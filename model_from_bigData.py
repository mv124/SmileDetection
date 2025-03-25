
# import the necessary packages
import os
import cv2
import numpy as np
from keras.models import Model
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input, BatchNormalization, Dropout
from keras.optimizers import Adam
import tensorflow as tf

# Load the pre-trained Haar cascade for face detection
face_cascade = cv2.CascadeClassifier('C:/Users/jania/OneDrive/Desktop/MEGHA/MEGHA_projects/'
                                     'My datasets/haarcascade_frontalface_default.xml')

# Path to the folder containing images
folder_path_train_0 = 'C:/Users/jania/OneDrive/Desktop/MEGHA/MEGHA_projects/My datasets/big data/train_folder/0'
folder_path_train_1 = 'C:/Users/jania/OneDrive/Desktop/MEGHA/MEGHA_projects/My datasets/big data/train_folder/1'
folder_path_test_0 = 'C:/Users/jania/OneDrive/Desktop/MEGHA/MEGHA_projects/My datasets/big data/test_folder/0'
folder_path_test_1 = 'C:/Users/jania/OneDrive/Desktop/MEGHA/MEGHA_projects/My datasets/big data/test_folder/1'


def load_images(folder_path):
    X = []

    for filename in os.listdir(folder_path):

        # Load image
        image_path0 = os.path.join(folder_path, filename)
        image = cv2.imread(image_path0)

        # Convert the image to grayscale (face detection works on grayscale images)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Detect faces in the grayscale image
        faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        if len(faces) > 0:

            # Preprocess image (resize, normalize, etc.)
            image = cv2.resize(image, (64, 64))
            image = image / 255.0  # Normalize pixel values to range [0, 1]

            # Add image to X_train
            X.append(image)

    return np.array(X)

X_train0 = load_images(folder_path_train_0)
X_train1 = load_images(folder_path_train_1)
X_train = np.concatenate((X_train0, X_train1))
y_train = np.concatenate((np.array([0 for i in X_train0]), np.array([1 for j in X_train1])))

X_test0 = load_images(folder_path_test_0)
X_test1 = load_images(folder_path_test_1)
X_test = np.concatenate((X_test0, X_test1))
y_test = np.concatenate((np.array([0 for p in X_test0]), np.array([1 for q in X_test1])))

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
print(y_test.shape)

# Define input shape
input_shape = X_train.shape[1:]
print("input_shape: ", input_shape)


# Define input layer
input_layer = Input(shape=input_shape)

# Add subsequent layers with Batch Normalization and Dropout
conv1 = Conv2D(32, kernel_size=(3, 3), activation='relu')(input_layer)
conv1 = BatchNormalization()(conv1)
conv1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(64, kernel_size=(3, 3), activation='relu')(conv1)
conv2 = BatchNormalization()(conv2)
conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)
flatten = Flatten()(conv2)
dense1 = Dense(256, activation='relu')(flatten)
dense1 = Dropout(0.1)(dense1)
dense2 = Dense(128, activation='relu')(dense1)
dense2 = Dropout(0.1)(dense2)
output_layer = Dense(1, activation='sigmoid')(dense2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])


# Train the model with augmented data
model.fit(X_train, y_train, epochs=16, batch_size=32)

# Evaluate the model
predictions = model.evaluate(x=X_test, y=y_test)
print("\nLoss:", predictions[0])
print("Test Accuracy:", predictions[1])


# Save the model
model.save('smile_detection_model_bigData.h5')

