import os
import numpy as np
from PIL import Image
import cv2 as cv
import pickle

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, 'images')

face_cascade = cv.CascadeClassifier('Data/haarcascade_frontalface_alt2.xml')
recognizer = cv.face.LBPHFaceRecognizer_create()


def image_train():
    current_id = 0
    label_ids = {}
    y_train = []
    x_train = []
    count = 0
    for root, dirs, files in os.walk(image_dir):

        for file in files:
            count += 1
            print(f'\rProgress - {count / 2057:.2%}', end='', flush=True)
            if file.endswith('jpg'):
                path = os.path.join(root, file)
                label = os.path.basename(root).replace(' ', '-').lower()

                if label not in label_ids:
                    label_ids[label] = current_id
                    current_id += 1
                img_id = label_ids[label]

                pil_image = Image.open(path).convert('L')  # Converts to greyscale
                final_image = pil_image.resize((178, 218), Image.ANTIALIAS)
                image_array = np.array(final_image, 'uint8')

                faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.1, minNeighbors=3)

                for (x, y, w, h) in faces:
                    region = image_array[y:y + h, x:x + w]
                    x_train.append(region)
                    y_train.append(img_id)

    with open('labels.pickle', 'wb') as f:
        pickle.dump(label_ids, f)
    recognizer.train(x_train, np.array(y_train))
    recognizer.save('trainer.yml')
