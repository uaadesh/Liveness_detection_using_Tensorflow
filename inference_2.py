import tensorflow as tf
import cv2
import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Load the saved model
model = tf.keras.models.load_model("D:\\Final year project\\Training\\training_3\\saved_model_3\\")

# Create a video capture object
cap = cv2.VideoCapture(1)

# Loop over frames
while True:
    # Capture the frame
    ret, frame = cap.read()
    if frame is None:
        continue
    # Convert the frame to a NumPy array
    frame = np.array(frame)

    # Resize the frame to the input size of the model
    frame = cv2.resize(frame, (224, 224))
    frame = np.expand_dims(frame, axis=0)
    print(frame.shape)

    # Make a prediction
    prediction = model.predict(frame)

    # Get the class label with the highest probability
    class_label = np.argmax(prediction)

    # Display the prediction
    cv2.putText(frame, "Prediction: {}".format(class_label), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Display the frame
    cv2.imshow("Prediction", frame)

    # Wait for a key press
    key = cv2.waitKey(1)

    # If the `q` key is pressed, break from the loop
    if key == ord("q"):
        break

# Release the capture object
cap.release()

# Close all windows
cv2.destroyAllWindows()
