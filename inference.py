import tensorflow as tf
import numpy as np
import cv2

# Establishing the connection with arduino

"""
import serial
import time
arduino = serial.Serial(port='COM12', baudrate=115200, timeout=.1)
def write_read(x):
    arduino.write(bytes(x, 'utf-8'))
    time.sleep(0.05)
    data = arduino.readline()
    return data
"""

# Load the model
model = tf.keras.models.load_model("D:\\Final year project\\Training\\training_vgg\\saved_model")

# Create a VideoCapture object for the camera
cap = cv2.VideoCapture(1)

while True:

    # Capture a frame from the camera
    ret, frame = cap.read()
    if frame is None:
        continue
    # Classify the image
    image = cv2.resize(frame, (224, 224))
    image = image.reshape(1, 224, 224, 3)
    #image = np.expand_dims(frame, axis=0)
    #print(image.shape)

    #image = tf.image.rgb_to_grayscale(frame)
    #image = tf.keras.preprocessing.image.img_to_array(image)
    #image = tf.keras.preprocessing.image.smart_resize(image, (224, 224))
    #image = image.reshape(1, 224, 224, 3)

    prediction = model.predict(image)
    class_label = np.argmax(prediction)

    # Display the classification result
    #print(prediction)
    text = "Real" if prediction[0][0]<=prediction[0][1] else "Fake"
    cv2.putText(frame, text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow("Image", frame)

    # Sending predictions to arduino
    # result = "1\n" if prediction[0][0]<=prediction[0][1] else "0\n"
    # result = write_read(result)
    # print(result) # printing the value

    # Wait for a key press
    key = cv2.waitKey(1)

    # If the key press is ESC, break the loop
    if key == 27:
        break

# Release the VideoCapture object
cap.release()

# Close all windows
cv2.destroyAllWindows()
