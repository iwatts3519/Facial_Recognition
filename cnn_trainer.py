import os

# Ensures that debug warning and information messages from Tensor Flow are suppressed when the program runs - this
# line has to run before Tensor Flow is imported
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # or any {'0', '1', '2'}
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D
from tensorflow.keras import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# The following code ensures errors with my graphics card are dealt with - an NVIDIA RTX3080. This might need
# commenting out to run on a cpu
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def train_cnn():
    # Image generators are used to preprocess images ready for processing - here we are just using the rescale function
    # to ensure that all elements of the tensor are between 0 and 1
    training_generator = ImageDataGenerator(rescale=1. / 255)
    testing_generator = ImageDataGenerator(rescale=1. / 255)

    # The two directory paths are assigned to strings
    training_directory = 'images/Train/'
    testing_directory = 'images/Test'

    # training data objects are set up for batch processing in sizes of 32
    training_data = training_generator.flow_from_directory(training_directory,
                                                           batch_size=32,
                                                           target_size=(178, 218),
                                                           class_mode='binary',
                                                           seed=0)
    testing_data = testing_generator.flow_from_directory(testing_directory,
                                                         batch_size=32,
                                                         target_size=(178, 218),
                                                         class_mode='binary',
                                                         seed=0)
    model = Sequential([
        Conv2D(10, 3, activation='relu', input_shape=(178, 218, 3)),
        MaxPool2D(pool_size=2),  # reduce number of features by half
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(),
        Conv2D(10, 3, activation='relu'),
        MaxPool2D(),
        Flatten(),
        Dense(1, activation='sigmoid')
    ])

    model.compile(loss='binary_crossentropy',
                  optimizer=Adam(),
                  metrics=['accuracy'])

    model.fit(training_data,
              epochs=5,
              steps_per_epoch=len(training_data),
              validation_data=testing_data,
              validation_steps=len(testing_data))
    model.save('Saved_CNN')
    model.summary()
    return model


def predict_image(model, image, class_names):
    tensor_image = tf.convert_to_tensor(image)
    resized_image = tf.image.resize(tensor_image, [178, 218])
    normalised_image = resized_image / 255.
    expanded_img_array = tf.expand_dims(normalised_image, axis=0)
    prediction = model.predict(expanded_img_array)
    return class_names[int(tf.round(prediction)[0][0])]
