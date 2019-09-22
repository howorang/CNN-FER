import datetime
import os

import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img
from keras_preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from tensorflow import keras

from serialization import load_dataset

IMAGE_PATH = "data/Emotion"
OUTPUT_PATH = 'data/output/'

def create_model():
    initial_size = 32
    return keras.Sequential([
        keras.layers.Conv2D(filters=initial_size,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu',
                            input_shape=(192, 192, 1)),
        keras.layers.Conv2D(filters=initial_size,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu'),
        keras.layers.Conv2D(filters=initial_size,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2),
                                  padding='valid'),
        keras.layers.Conv2D(filters=initial_size * 2,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu'),
        keras.layers.Conv2D(filters=initial_size * 2,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu'),
        keras.layers.Conv2D(filters=initial_size * 2,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu'),
        keras.layers.MaxPooling2D(pool_size=(2, 2),
                                  padding='valid'),
        keras.layers.Conv2D(filters=initial_size * 4,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu'),
        keras.layers.Conv2D(filters=initial_size * 4,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu'),
        keras.layers.Conv2D(filters=initial_size * 4,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu'),

        keras.layers.MaxPooling2D(pool_size=(2, 2),
                                  padding='valid'),
        keras.layers.Conv2D(filters=initial_size * 8,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu'),
        keras.layers.Conv2D(filters=initial_size * 8,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu'),
        keras.layers.Conv2D(filters=initial_size * 8,
                            kernel_size=(3, 3),
                            strides=(1, 1),
                            activation='relu'),

        keras.layers.MaxPooling2D(pool_size=(2, 2),
                                  strides=(2, 2),
                                  padding='valid'),

        # first flatten
        keras.layers.GlobalAveragePooling2D(),
        keras.layers.Flatten(),
        keras.layers.Dense(8, activation=tf.nn.softmax)
    ])


labels, images = load_dataset('example20190922183246')
# if 1 channel dataset

images, X_test, labels, y_test = train_test_split(images, labels, test_size=0.15)

# labels = np.repeat(labels, 10, axis=0)
# images = np.repeat(images, 10, axis=0)

batch_size = 20

logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
print(logdir)

model = create_model()
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(x=images,
                    y=labels,
                    validation_split=0.15,
                    epochs=90,
                    verbose=1,
                    batch_size=32,
                    shuffle=True,
                    callbacks=[tensorboard_callback])

test_scalar_loss = model.evaluate(
    x=X_test,
    y=y_test,
    batch_size=32,
    verbose=1
)
