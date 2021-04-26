import cv2 as cv
import pickle


def test_pretrained():
    print("\n1. Proceeding with running Open CV model with webcam")

    print('2. Setting Up Cascade Classifier')
    face_cascade = cv.CascadeClassifier('Data/haarcascade_frontalface_default.xml')
    print('3. Setting Up Recognizer')
    recognizer = cv.face.LBPHFaceRecognizer_create()
    print('4. Loading in Trained YAML file - may take a few minutes')
    recognizer.read('trainer.yml')

    print('5. Loading Labels in from PICKLE file')
    with open('labels.pickle', 'rb') as f:
        original_labels = pickle.load(f)
    labels = {v: k for (k, v) in original_labels.items()}

    print('6. Setting Up Webcam')
    stream = cv.VideoCapture(1)
    window_name = "Stream - Press esc to Exit"

    print('7. Beginning Loop for Webcam (Webcam Window may not have focus - check taskbar for webcam window)')

    while True:
        # Read in the frame from the stream - the frame is the second part in the tuple and is the only part we are
        # interested in
        _, frame = stream.read()
        # Convert to greyscale - mimics the training of the model which is more straightforward if colour is removed
        grayscale = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # Set up the Face Cascade Object
        faces = face_cascade.detectMultiScale(grayscale, scaleFactor=1.1, minNeighbors=3, minSize=(20, 20))
        for (x, y, w, h) in faces:
            # Extract the Region of Interest (ROI) and draw a blue box around it
            roi_gray = grayscale[y:y + h, x:x + w]
            color = (255, 0, 0)
            l_width = 2
            cv.rectangle(frame, (x, y), (x + w, y + h), color, l_width)

            # Predict the gender using the recognizer and insert a label onto the frame
            img_id, _ = recognizer.predict(roi_gray)
            name = f'{labels[img_id]}'
            font = cv.FONT_HERSHEY_SIMPLEX
            text_colour = (255, 255, 255)
            text_width = 2
            cv.putText(frame, name, (x, y), font, 1, text_colour, text_width, cv.LINE_AA)

            # Display the window with the frame including the text label on it
            cv.imshow(window_name, frame)
        # Wait for 30 milliseconds and if the escape key is pressed then break out of the while loop
        k = cv.waitKey(30)
        if k == 27:
            break

    stream.release()
    cv.destroyAllWindows()
