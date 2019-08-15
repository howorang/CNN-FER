import numpy as np
import tensorflow as tf
from hyperas import optim
from hyperas.distributions import choice
from hyperopt import Trials, STATUS_OK, tpe
from sklearn.model_selection import train_test_split
from tensorflow import keras

from data_loader import get_dataset


def create_model(X_train, Y_train, X_test, Y_test):
    initial_size = {{choice([12, 32, 64, 128])}}
    model = keras.Sequential([
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
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    result = model.fit(X_train,
                       Y_train,
                       validation_split=0.1,
                       epochs=4,
                       verbose=1,
                       batch_size=32,
                       shuffle=True)

    print(result.history)
    validation_acc = np.amax(result.history['val_accuracy'])
    return {'loss': -validation_acc, 'status': STATUS_OK, 'model': model}


def data():
    labels, images = get_dataset()
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.15)
    X_train = np.repeat(X_train, 10, axis=0)
    Y_train = np.repeat(Y_train, 10, axis=0)
    return X_train, Y_train, X_test, Y_test


if __name__ == '__main__':
    best_run, best_model = optim.minimize(model=create_model,
                                          data=data,
                                          algo=tpe.suggest,
                                          max_evals=5,
                                          trials=Trials())
    X_train, Y_train, X_test, Y_test = data()
    print("Evalutation of best performing model:")
    print(best_model.evaluate(X_test, Y_test))
    print("Best performing model chosen hyper-parameters:")
    print(best_run)
