import cv2 as cv
import pathlib
import cnn_trainer as cnn
import tensorflow as tf
import numpy as np


def cnn_predict():
    print("\n1. Proceeding with running CNN model with webcam")

    print('2. Setting Up Cascade Classifier')
    face_cascade = cv.CascadeClassifier('Data/haarcascade_frontalface_default.xml')

    print('3. Setting Up Webcam')
    stream = cv.VideoCapture(1)
    window_name = "Stream - Press esc to Exit"

    print('4. Beginning Loop for Webcam (Webcam Window may not have focus - check taskbar for webcam window)')

    data_dir = pathlib.Path('images/Train/')  # turn our training path into a Python path
    class_names = np.array(
        sorted([item.name for item in data_dir.glob('*')]))  # created a list of class_names from the subdirectories
    while True:
        ret, frame = stream.read()
        grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        for (x, y, w, h) in faces:
            # Load in the pretrained CNN model
            model = tf.keras.models.load_model('Saved_CNN')
            tensor_image = tf.convert_to_tensor(frame)
            name = cnn.predict_image(model, tensor_image, class_names=class_names)
            color = (255, 0, 0)
            l_width = 2
            cv.rectangle(frame, (x, y), (x + w, y + h), color, l_width)

            font = cv.FONT_HERSHEY_SIMPLEX

            text_colour = (255, 255, 255)
            text_width = 2
            cv.putText(frame, name, (x, y), font, 1, text_colour, text_width, cv.LINE_AA)

            cv.imshow(window_name, frame)

        k = cv.waitKey(30)
        if k == 27:
            break

    stream.release()
    cv.destroyAllWindows()
