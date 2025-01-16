import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained CNN model
model = tf.keras.models.load_model('/Users/aman/Documents/Work/Machine Learning/Rock-Paper-Scissors-Using-ML/Models/Model.h5')

# Open the camera
cap = cv2.VideoCapture(0)
cv2.namedWindow("Real-time CNN Classification", cv2.WINDOW_FULLSCREEN)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    # Preprocess the frame:
    resizedFrame = cv2.resize(frame, (128, 128))
    normalizedFrame = resizedFrame / 255.0

    # Add batch dimension (1, 128, 128, 3)
    inputFrame = np.expand_dims(normalizedFrame, axis=0)

    # Make prediction
    prediction = model.predict(inputFrame)
    predictedClass = np.argmax(prediction, axis=-1)

    # Get the class label 
    classLabels = ['Paper', 'Rock', 'Scissors'] 
    predictedLabel = classLabels[predictedClass[0]]

    # Display the result on the frame
    cv2.putText(frame, f"Prediction: {predictedLabel}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 0), 10)

    # Show the frame with the prediction
    cv2.imshow('Real-time CNN Classification', frame)

    # Press 'q' to exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()