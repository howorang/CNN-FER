import datetime
import os
from enum import Enum

import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from keras.preprocessing.image import load_img
from keras_preprocessing.image import load_img
from sklearn.model_selection import train_test_split
from tensorflow import keras
import json
from keras.utils import to_categorical

from resnet import resnet50
from serialization import load_dataset
from vgg19 import vgg19


class DatasetFiles(Enum):
    RAFD = 'rafd'
    CK = 'ck'
    JAFFE = 'jaffe'
    HETERO = 'hetero'

class Network(Enum):
    VGG19 = vgg19()
    RESNET = resnet50()

class Optimizer(Enum):
    DEFAULT = keras.optimizers.Adam()
    LR = keras.optimizers.Adam(lr=0.00001)

testing_scenerios = [
    # (Network.RESNET, DatasetFiles.RAFD, Optimizer.DEFAULT),
    # (Network.RESNET, DatasetFiles.RAFD, Optimizer.LR),
    # (Network.VGG19, DatasetFiles.RAFD, Optimizer.DEFAULT),

    # (Network.RESNET, DatasetFiles.JAFFE, Optimizer.DEFAULT),
    # (Network.VGG19, DatasetFiles.JAFFE, Optimizer.DEFAULT),
    # (Network.VGG19, DatasetFiles.JAFFE, Optimizer.LR),

    # (Network.RESNET, DatasetFiles.HETERO, Optimizer.DEFAULT),
    (Network.RESNET, DatasetFiles.HETERO, Optimizer.LR),
    (Network.VGG19, DatasetFiles.HETERO, Optimizer.DEFAULT),
    (Network.VGG19, DatasetFiles.HETERO, Optimizer.LR),


]

for scenerio in testing_scenerios:
    print(scenerio[0].name + scenerio[1].name + scenerio[2].name)
    dataset = scenerio[1]
    network = scenerio[0]
    optimizer = scenerio[2]


    labels, images = load_dataset(dataset.value + '3000.h5')
    # if 1 channel dataset

    labels = [to_categorical(i - 1, 6) for i in labels]
    labels = np.array(labels)

    if len(images.shape) == 3:
        resotred_shape = images.shape + (1,)
        images = np.reshape(images, resotred_shape)
    # if 1 channel dataset



    images, X_test, labels, y_test = train_test_split(images, labels, test_size=0.15)

    labels = np.repeat(labels, 4, axis=0)
    images = np.repeat(images, 4, axis=0)

    batch_size = 20

    logdir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S_" + network.name + "_" + dataset.name + "_" + optimizer.name))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(logdir, histogram_freq=1)
    print(logdir)

    model = network.value
    model.compile(optimizer=optimizer.value,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    history = model.fit(x=images,
                        y=labels,
                        validation_split=0.15,
                        epochs=30,
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
    model.save(os.path.join(logdir + os.path.sep + "model"))
    print(str(test_scalar_loss),  file=open(os.path.join(logdir + os.path.sep + "scalar.txt"), "a"))


