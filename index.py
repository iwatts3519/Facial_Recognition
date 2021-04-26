import opencv_train_images as ti
import opencv_pretrained as ocv
import cnn_trainer as cnn
import cnn_pretrained as cnp

while True:
    choice = input("Which Model would you like to test: \n1. OpenCV Classifier\n2. CNN Classifier\n Please type a 1 "
                   "or a 2, or q to quit")
    if choice == '1':
        print("OpenCV Classifier")
        choice1 = input("Would you like to retrain the model? (y/n)")
        if choice1 == 'y':
            print("Model Being Retrained")
            ti.image_train()
        ocv.test_pretrained()
    elif choice == '2':
        print("CNN Classifier")
        choice1 = input("Would you like to retrain the model? (y/n)")
        if choice1 == 'y':
            print("Model Being Retrained")
            cnn.train_cnn()
        cnp.cnn_predict()
    elif choice == 'q':
        break
